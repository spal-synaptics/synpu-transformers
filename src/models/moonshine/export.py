import argparse
import json
import os
from math import floor
from pathlib import Path
from subprocess import check_output, CalledProcessError, STDOUT
from typing import Literal, Final

import onnx
import onnx_graphsurgeon as gs
import numpy as np
from datasets import load_dataset, Audio
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoProcessor
from onnxruntime.transformers.optimizer import optimize_model

from . import (
    DEFAULT_INPUT_AUDIO_S,
    DEFAULT_DEC_TOK_PER_SEC,
    DEFAULT_MODEL_SIZE,
    ONNX_DTYPES,
    OPTIMUM_DTYPES,
    STATIC_MODEL_COMPONENTS,
)
from ._graph import *
from ._inference import MoonshineDynamic, MoonshineStatic
from ...utils.onnx import (
    get_model_ops_count,
    print_onnx_model_inputs_outputs_info,
    add_onnx_args,
    check_dynamic_shapes,
)
from ...utils.synpu import (
    export_iree
)


class MoonshineModelExporter:

    COMPONENTS: Final[dict[str, str]] = {
        "encoder": "encoder_model.onnx",
        "decoder": "decoder_model.onnx",
        "decoder_with_past": "decoder_with_past_model.onnx",
    }

    COMPONENTS_MERGED: Final[dict[str, str]] = {
        "encoder": "encoder_model.onnx",
        "decoder_merged": "decoder_model_merged.onnx",
    }

    def __init__(
        self,
        model_size: Literal["base", "tiny"] = "tiny",
        model_dtype: str = "float",
        static_models: bool = True,
        *,
        max_audio_s: int = 5,
        max_tok_per_s: int = 6,
        models_dir: str | os.PathLike = "models/onnx",
        show_model_info: bool = False,
        skip_optimum: bool = False,
    ):
        if model_size not in ["base", "tiny"]:
            raise ValueError(
                f"Invalid model size '{model_size}', choose one of: ['base', 'tiny']"
            )

        self._model_size = model_size
        self._model_dtype = model_dtype
        self._static_models = static_models
        self._models_dir = Path(models_dir)
        self._show_model_info = show_model_info
        self._hf_repo = "UsefulSensors/moonshine"
        self._config = AutoConfig.from_pretrained(f"{self._hf_repo}-{self._model_size}")
        self._num_samples = max_audio_s * 16_000
        self._max_tokens = max_audio_s * max_tok_per_s
        self._enc_seq_len = (
            floor(floor(floor(self._num_samples / 64 - 127 / 64) / 3) / 2) - 1
        )

        if not skip_optimum:
            self._model_dtype = "fp32" if self._model_dtype == "float" else self._model_dtype
            if self._model_dtype not in OPTIMUM_DTYPES:
                raise ValueError(f"'{self._model_dtype}' is an invalid dtype for optimium export, choose one of {OPTIMUM_DTYPES}")
            self._onnx_dir = self._models_dir / self._hf_repo / "onnx" / self._model_size / self._model_dtype
            self._onnx_dir.mkdir(parents=True, exist_ok=True)
            self._optimum_export_models()
        else:
            if self._model_dtype not in ONNX_DTYPES:
                raise ValueError(f"'{self._model_dtype}' is an invalid dtype for pre-existing ONNX models, choose one of {ONNX_DTYPES}")
            self._onnx_dir = self._models_dir / self._hf_repo / "onnx" / "merged" / self._model_size / self._model_dtype
            self._onnx_dir.mkdir(parents=True, exist_ok=True)
            self._hf_download_models()

        self._components, self._merged_decoder = self._load_onnx()
        self._export_dir = (
            self._models_dir
            / self._hf_repo
            / "export"
            / self._model_size
            / self._model_dtype
        )
        self._export_dir.mkdir(parents=True, exist_ok=True)
        self._export_paths: dict[str, Path] = {}

    @staticmethod
    def check_model(model: onnx.ModelProto, skip_data_prop: bool = False) -> onnx.ModelProto:
        if model.ir_version > 10:
            print(
                f"Warning: Model IR version is > 10 ({model.ir_version}), which might be unsupported by onnxruntime"
            )
        model = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True, data_prop=not skip_data_prop
        )
        onnx.checker.check_model(model, full_check=True)
        return model

    @staticmethod
    def split_merged_decoder(merged_model: onnx.ModelProto) -> tuple[onnx.ModelProto, onnx.ModelProto]:
        assert merged_model.ir_version <= 10
        if_node = next(n for n in merged_model.graph.node if n.op_type == "If")
        then_branch = None
        else_branch = None
        for attr in if_node.attribute:
            if attr.name == "then_branch":
                then_branch = attr.g
            elif attr.name == "else_branch":
                else_branch = attr.g
        if not then_branch or not else_branch:
            raise ValueError("Merged decoder If node missing branches")
        
        outputs = merged_model.graph.output
        same_outputs: bool = all([
            out_merged == out == out_with_past 
            for out_merged, out, out_with_past 
            in zip(
                [out.name for out in outputs],
                [out.name for out in then_branch.output],
                [out.name for out in else_branch.output]
            )
        ])

        decoder_graph = onnx.helper.make_graph(
            nodes=else_branch.node,
            name="main_graph",
            inputs=[input for input in merged_model.graph.input if input.name in ("input_ids", "encoder_hidden_states")],
            outputs=outputs if same_outputs else else_branch.output,
            initializer=list(merged_model.graph.initializer) + list(else_branch.initializer)
        )
        decoder_model = onnx.helper.make_model(decoder_graph, opset_imports=merged_model.opset_import)
        decoder_model.ir_version = merged_model.ir_version

        decoder_with_past_graph = onnx.helper.make_graph(
            nodes=then_branch.node,
            name="main_graph",
            inputs=[input for input in merged_model.graph.input if input.name not in ("encoder_hidden_states", "use_cache_branch")],
            outputs=[out for out in (outputs if same_outputs else then_branch.output) if "encoder" not in out.name],
            initializer=list(merged_model.graph.initializer) + list(then_branch.initializer)
        )
        decoder_with_past_model = onnx.helper.make_model(decoder_with_past_graph, opset_imports=merged_model.opset_import)
        decoder_with_past_model.ir_version = merged_model.ir_version

        return decoder_model, decoder_with_past_model

    def _optimum_export_models(self):
        if not all(
            (self._onnx_dir / comp_model_name).exists()
            for comp_model_name in MoonshineModelExporter.COMPONENTS.values()
        ):
            try:
                check_output(
                    [
                        "optimum-cli", "export", "onnx",
                        str(self._onnx_dir),
                        "--model", f"{self._hf_repo}-{self._model_size}",
                        "--dtype", self._model_dtype,
                        "--opset", "17",
                    ],
                    text=True,
                    stderr=STDOUT,
                )
            except CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to export ONNX model via '{' '.join(e.cmd)}':\n    "
                    + "\n    ".join(e.output.strip().splitlines())
                ) from None
            for comp, comp_model_name in MoonshineModelExporter.COMPONENTS.items():
                self.optimize_model(self._onnx_dir / comp_model_name, comp)

    def _hf_download_models(self):
        for comp_model_name in MoonshineModelExporter.COMPONENTS_MERGED.values():
            hf_hub_download(
                self._hf_repo,
                comp_model_name,
                subfolder=f"onnx/merged/{self._model_size}/{self._model_dtype}",
                local_dir=self._models_dir / self._hf_repo,
            )

    def _load_onnx(self) -> tuple[dict[str, onnx.ModelProto], bool]:
        unmerged_model_names: set[str] = set(self.COMPONENTS.values())
        merged_model_names: set[str] = set(self.COMPONENTS_MERGED.values())
        model_names: set[str] = set(list(p.name for p in self._onnx_dir.glob("*.onnx")))
        if model_names == unmerged_model_names:
            print(f"[ONNX-load] Found encoder and un-merged decoder models @ '{self._onnx_dir}'")
            merged_decoder = False
        elif model_names == merged_model_names:
            print(f"[ONNX-load] Found encoder and merged decoder model @ '{self._onnx_dir}'")
            merged_decoder = True
        else:
            raise ValueError(
                f"Expected merged models {merged_model_names} or un-merged models {unmerged_model_names} @ '{self._onnx_dir}'"
            )
        comps = MoonshineModelExporter.COMPONENTS_MERGED if merged_decoder else MoonshineModelExporter.COMPONENTS
        onnx_models: dict[str, onnx.ModelProto] = {
            comp: onnx.load(self._onnx_dir / comp_model_name)
            for comp, comp_model_name in comps.items()
        }
        return onnx_models, merged_decoder

    def _make_encoder_model_static(
        self, batch_dim: str, inp_len_dim: str, enc_seq_len_dims: list[str]
    ) -> onnx.ModelProto:
        """
        Make the encoder model static by replacing dynamic dimensions with fixed values.

        Args:
            batch_dim (str): Name of dynamic batch dimension to replace with 1.
            inp_len_dim (str): Name of the input length dimension to replace with `self.num_samples`

        Returns:
            onnx.ModelProto: The modified encoder model with static dimensions

        Raises:
            ValueError: If an unexpected dynamic dimension is found in the model inputs or outputs
        """

        graph: gs.Graph = gs.import_onnx(self._components["encoder"])
        for tensor in graph.inputs + graph.outputs:
            old_shape = list(tensor.shape)
            for i, d in enumerate(tensor.shape):
                if not isinstance(d, str):
                    continue
                if d.isdigit():
                    tensor.shape[i] = int(d)
                elif d == batch_dim:
                    tensor.shape[i] = 1
                elif d == inp_len_dim:
                    tensor.shape[i] = self._num_samples
                elif d in enc_seq_len_dims:
                    tensor.shape[i] = self._enc_seq_len
                else:
                    raise ValueError(
                        f"Unexpected dynamic dimension '{d}' in tensor '{tensor.name}'"
                    )
            print(
                f"[encoder] Fixing IO dims '{tensor.name}': {old_shape} -> {tensor.shape}"
            )

        graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()
        new_encoder = onnx.shape_inference.infer_shapes(
            gs.export_onnx(graph), check_type=True, strict_mode=True, data_prop=True
        )
        new_encoder.ir_version = self._components["encoder"].ir_version
        return new_encoder

    def _make_decoder_model_static(
        self, decoder_model: onnx.ModelProto, with_past: bool
    ) -> onnx.ModelProto:
        """
        Make decoder models static by replacing dynamic dimensions with fixed values and applying necessary transformations.

        Replaces KV caching and other dynamic operations with static equivalents in the cached decoder model.

        Args:
            decoder_model (onnx.ModelProto): ONNX decoder model to modify
            with_past (bool): Whether the model is the cached branch of the decoder

        Returns:
            onnx.ModelProto: The modified decoder model with static dimensions and transformations applied

        Raises:
            ValueError: If an unexpected dynamic dimension is found in the model inputs, outputs, or nodes
        """
        graph: gs.Graph = gs.import_onnx(decoder_model)
        output_names = {o.name for o in graph.outputs}
        comp = "decoder" + ("_with_past" if with_past else "")
        pad_len = (
            self._config.hidden_size // self._config.decoder_num_attention_heads
        ) % 8

        for tensor in graph.inputs + graph.outputs:
            old_shape = list(tensor.shape)
            for i, d in enumerate(tensor.shape):
                if not isinstance(d, str):
                    continue
                if d.isdigit():
                    tensor.shape[i] = int(d)
                elif "past_decoder_sequence_length" in d:
                    tensor.shape[i] = self._max_tokens if with_past else 1
                elif "encoder_sequence_length" in d:
                    tensor.shape[i] = self._enc_seq_len
                elif d == "decoder_sequence_length":
                    tensor.shape[i] = 1
                elif d == "batch_size":
                    tensor.shape[i] = 1
                else:
                    raise ValueError(
                        f"Unexpected dynamic dimension '{d}' in tensor '{tensor.name}'"
                    )
            print(
                f"[{comp}] Fixing IO dims '{tensor.name}': {old_shape} -> {tensor.shape}"
            )

        for node in graph.nodes:

            # Remove or replace isNaN ops
            if node.op == "IsNaN":
                graph.remove_is_NaN(node)
                print(f"[{comp}] Removed unsupported IsNaN op '{node.name}'")

            # Move model output if it's fed by a Concat node which has a Pad consumer
            if (
                not with_past
                and node.op == "Concat"
                and node.outputs[0].name in output_names
            ):
                output_name = node.outputs[0].name
                consumers: list[gs.Node] = list(node.outputs[0].outputs)
                for consumer in consumers:
                    if consumer.op == "Pad":
                        graph.move_output_from_concat_to_slice(node, consumer, pad_len)
                        print(
                            f"[{comp}] Moved output '{output_name}' to Pad node '{consumer.name}'"
                        )

        # TODO: Add Cast nodes before activation ops
        if with_past:
            cur_len_2d = gs.Variable("current_len", dtype=np.int64, shape=[1, 1])
            graph.inputs.append(cur_len_2d)
            cur_len = graph.layer(
                name="current_len_to_1d",
                op="Squeeze",
                inputs=[cur_len_2d, [0]],
                outputs=[
                    gs.Variable(
                        cur_len_2d.name + "_squeezed", dtype=np.int64, shape=[1]
                    )
                ],
            )[0]

            for node in graph.nodes:

                # Replace dynamic KV cache update
                if node.op == "Concat" and node.outputs[0].name in output_names:
                    cache_output = node.outputs[0].name
                    if node.attrs["axis"] != -2:
                        raise ValueError(
                            f"Static KV Cache: '{node.name}' expected Concat axis to be -2, got {node.attrs['axis']}"
                        )
                    if len(node.inputs) != 2:
                        raise ValueError(
                            f"Static KV Cache: '{node.name}' expected Concat node to have 2 inputs, got {len(node.inputs)}"
                        )
                    graph.replace_kv_concat_with_mask(node, cur_len, self._max_tokens)
                    print(f"[{comp}] Added static KV cache for output '{cache_output}'")

                # Add causal attention score mask
                if node.op == "Softmax" and node.name.endswith("self_attn/Softmax"):
                    if (prod := node.i()).op != "Add":
                        raise ValueError(
                            f"Causal Attention Mask: '{node.name}' expected producer to be Add node, got {prod.op} ({prod.name})"
                        )
                    graph.add_causal_attn_score_mask(prod, cur_len, self._max_tokens)
                    print(
                        f"[{comp}] Added causal attention mask to scores at node '{node.name}'"
                    )

                # Replace dynamic sequence length getter with `cur_len`
                if node.op == "Shape" and "past_key_values" in node.inputs[0].name:
                    graph.replace_dynamic_seq_len_getter(node, cur_len)
                    print(
                        f"[{comp}] Replaced dynamic seq len getter at node '{node.name}'"
                    )

                # Replace dynamic index computation `Range(start, start + 1, 1) -> index`
                if (
                    node.op == "Range"
                    and node.i(1).op == "Add"
                    and any(inp is node.inputs[0] for inp in node.i(1).inputs)
                ):
                    graph.replace_dynamic_range_index(node)
                    print(
                        f"[{comp}] Replaced dynamic range index for node '{node.name}'"
                    )

        graph = graph.cleanup(
            remove_unused_graph_inputs=True, remove_unused_node_outputs=True
        ).toposort()
        new_model = onnx.shape_inference.infer_shapes(
            gs.export_onnx(graph), check_type=True, strict_mode=True, data_prop=True
        )
        new_model.ir_version = decoder_model.ir_version
        return new_model

    def make_static(
        self,
        *,
        batch_dim: str = "batch_size",
        inp_len_dim: str = "num_samples",
        enc_seq_len_dims: list[str] = ["encoder_sequence_length", "floor(floor(floor(num_samples/64 - 127/64)/3)/2) - 1"],
    ):
        if self._merged_decoder:
            print(f"[decoder_merged] Splitting merged decoder ...")
            decoder, decoder_with_past = self.split_merged_decoder(self._components["decoder_merged"])
            self._components["decoder"] = self.check_model(decoder)
            self._components["decoder_with_past"] = self.check_model(decoder_with_past)
            del self._components["decoder_merged"]
            assert set(self._components) == set(STATIC_MODEL_COMPONENTS)
            print(f"[decoder_merged] Decoder split into regular and with_past models")

        self._components["encoder"] = self._make_encoder_model_static(
            batch_dim, inp_len_dim, enc_seq_len_dims
        )
        self._components["decoder"] = self._make_decoder_model_static(
            self._components["decoder"], False
        )
        self._components["decoder_with_past"] = self._make_decoder_model_static(
            self._components["decoder_with_past"], True
        )

    def optimize_model(self, model_path: str | os.PathLike, component: str):
        optimized = optimize_model(
            str(model_path),
            model_type="bert",
            num_heads=(
                self._config.encoder_num_attention_heads
                if "encoder" in component
                else self._config.decoder_num_attention_heads
            ),
            hidden_size=self._config.hidden_size,
            only_onnxruntime=True,
            verbose=False,
        )
        optimized.save_model_to_file(str(model_path))
        optimized_model = onnx.load(model_path)
        optimized_model = onnx.shape_inference.infer_shapes(
            optimized_model, check_type=True, strict_mode=True, data_prop=False
        )
        self.check_model(onnx.load(model_path), skip_data_prop="decoder" in component and self._merged_decoder)
        onnx.save(optimized_model, model_path)

    def export_onnx(self, validate: bool = True):
        if self._static_models:
            self.make_static()

        for comp, model in self._components.items():
            self._export_paths[comp] = self._export_dir / f"{comp}.onnx"
            print(f"[{comp}] Checking model...")
            model = self.check_model(model, skip_data_prop="decoder" in comp and self._merged_decoder)
            onnx.save(model, self._export_paths[comp])
            print(f"[{comp}] Optimizing model...")
            self.optimize_model(self._export_paths[comp], comp)
            if self._static_models:
                print(f"[{comp}] Verifying static shapes...")
                dynamic_shapes = check_dynamic_shapes(onnx.load(self._export_paths[comp]))
                if dynamic_shapes:
                    raise ValueError(
                        f"Model '{comp}' still has dynamic shapes: {json.dumps(dynamic_shapes)}"
                    )
            if self._show_model_info:
                print(f"\n\nInfo for model '{self._export_paths[comp]}':")
                print_onnx_model_inputs_outputs_info(self._export_paths[comp])
                print(f"\nModel ops summary:")
                print(
                    json.dumps(
                        get_model_ops_count(onnx.load(self._export_paths[comp])), indent=4
                    ),
                    end="\n\n",
                )
            print(f"[{comp}] Saved model to '{self._export_paths[comp]}'")

        if validate:
            self.validate_onnx()

    def validate_onnx(self, n_iters: int = 5):

        def _sample_input(idx: int) -> np.ndarray:
            sample = dataset[idx]["audio"]
            inputs: np.ndarray = processor(
                sample["array"],
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="np",
            )
            return inputs["input_values"]

        if self._static_models:
            runner = MoonshineStatic.from_onnx(
                encoder_model=self._export_dir / "encoder.onnx",
                decoder_model=self._export_dir / "decoder.onnx",
                decoder_with_past_model=self._export_dir / "decoder_with_past.onnx",
                model_size=self._model_size
            )
        else:
            runner = MoonshineDynamic.from_onnx(
                encoder_model=self._export_dir / "encoder.onnx",
                decoder_model=self._export_dir / "decoder_merged.onnx",
                model_size=self._model_size
            )
        val_runner = MoonshineDynamic.from_onnx(
            encoder_model=self._onnx_dir / "encoder_model.onnx",
            decoder_model=self._onnx_dir / "decoder_model_merged.onnx",
            model_size=self._model_size,
            max_inp_len=runner.max_inp_len
        )

        processor = AutoProcessor.from_pretrained(f"{self._hf_repo}-{self._model_size}")
        dataset = load_dataset(
            path="hf-internal-testing/librispeech_asr_dummy",
            name="clean",
            split="validation",
        )
        dataset = dataset.cast_column(
            "audio", Audio(processor.feature_extractor.sampling_rate)
        )
        print(f"[ONNX-validation] loaded dataset 'hf-internal-testing/librispeech_asr_dummy', details: {dataset}")

        for i in range(n_iters):
            if i >= len(dataset):
                print("[ONNX-validation] no more samples to validate, stopping")
                break

            input = _sample_input(i)
            tokens = runner.run(input)
            print(f"[ONNX-validation] (iter {i}, {runner.last_infer_time * 1000:.3f} ms): ", end="")
            val_tokens = val_runner.run(input)
            if not np.array_equal(tokens, val_tokens):
                print(f"Warning: Validation failed, mismatched outputs\nExpected:\n{val_tokens},\nGenerated:\n{tokens}")
            else:
                print(f"Validation successful, identical outputs")

    def export_iree(self, iree_dir: str | os.PathLike):
        for comp, export_path in self._export_paths.items():
            print(f"[IREE-export] Exporting {comp} model @ '{export_path}' to IREE...")
            self.check_model(onnx.load(export_path), skip_data_prop="decoder" in comp and self._merged_decoder)
            iree_model_path = Path(iree_dir) / "moonshine" / self._model_size / self._model_dtype
            export_iree(
                export_path,
                iree_model_path
            )
            print(f"[IREE-export] Successfully exported '{iree_model_path}/{export_path.stem}.vmfb'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Moonshine to SyNAP")
    parser.add_argument(
        "-i",
        "--input-seconds",
        type=int,
        default=DEFAULT_INPUT_AUDIO_S,
        help="Input audio length in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--tokens-per-sec",
        type=int,
        default=DEFAULT_DEC_TOK_PER_SEC,
        help="Max number of tokens decoded per second (default: %(default)d)",
    )
    parser.add_argument(
        "-s",
        "--model-size",
        type=str,
        choices=["base", "tiny"],
        default=DEFAULT_MODEL_SIZE,
        help="Moonshine model size to export (default: %(default)s)",
    )
    add_onnx_args(
        parser,
        model_dtypes=ONNX_DTYPES + OPTIMUM_DTYPES,
        allow_no_opt=False,
    )
    parser.add_argument(
        "--dynamic-models",
        action="store_true",
        default=False,
        help="Export dynamic models for CPU"
    )
    parser.add_argument(
        "--skip-optimum",
        action="store_true",
        default=False,
        help="Use pre-exported ONNX models instead of exporting with optimum-cli"
    )
    parser.add_argument(
        "--skip-iree",
        action="store_true",
        default=False,
        help="Skip exporting to IREE"
    )
    args = parser.parse_args()

    exporter = MoonshineModelExporter(
        args.model_size,
        args.dtype,
        not args.dynamic_models,
        max_audio_s=args.input_seconds,
        max_tok_per_s=args.tokens_per_sec,
        models_dir=args.onnx_dir,
        show_model_info=args.show_model_info,
        skip_optimum=args.skip_optimum
    )
    exporter.export_onnx()
    if not args.skip_iree:
        exporter.export_iree("models/iree")
