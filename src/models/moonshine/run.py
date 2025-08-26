import os
import argparse
import logging
from pathlib import Path

import soundfile as sf
import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from ._inference import MoonshineDynamic, MoonshineStatic
from ...utils.demo import (
    add_common_args,
    format_answer,
)
from ...utils.logging import (
    add_logging_args,
    configure_logging,
)
from ...inference.runners import IREEInferenceRunner


def _find_models(model_dir: str | os.PathLike) -> tuple[list[Path], str]:
    p = Path(model_dir)
    onnx = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".onnx")
    vmfb = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".vmfb")
    if onnx and vmfb:
        raise ValueError(
            "Model directory contains both ONNX and VMFB models; choose one format."
        )
    models = onnx or vmfb
    if not models:
        raise FileNotFoundError(f"No .onnx or .vmfb models found in {p}")
    kind = "onnx" if onnx else "vmfb"
    logger.debug(
        "Selected %s models (%d): %s", kind, len(models), ", ".join(map(str, models))
    )
    return models, kind


def _load_moonshine(
    model_dir: str | os.PathLike,
    model_size: str,
    max_inp_len: int | None,
    max_dec_len: int | None,
) -> MoonshineDynamic | MoonshineStatic:
    models, kind = _find_models(model_dir)
    encoder = None
    decoder_merged = None  # dynamic model
    decoder, decoder_with_past = None, None  # static model
    for m in models:
        if m.stem == "encoder":
            encoder = m
        elif m.stem == "decoder_merged":
            decoder_merged = m
        elif m.stem == "decoder":
            decoder = m
        elif m.stem == "decoder_with_past":
            decoder_with_past = m
    is_static = decoder_merged is None and decoder is not None and decoder_with_past is not None
    if not encoder:
        raise FileNotFoundError(
            f"Missing encoder model 'encoder.{kind}' @ '{model_dir}'"
        )
    if not decoder_merged and not is_static:
        raise FileNotFoundError(
            f"Missing merged decoder model 'decoder_merged.{kind}' @ '{model_dir}'"
        )
    if not decoder_merged:
        if not decoder:
            raise FileNotFoundError(
                f"Missing decoder model 'decoder.{kind}' @ '{model_dir}'"
            )
        if not decoder_with_past:
            raise FileNotFoundError(
                f"Missing decoder with past model 'decoder_with_past.{kind}' @ '{model_dir}'"
            )

    if is_static:
        if kind == "vmfb":
            if not isinstance(max_inp_len, int) or not isinstance(max_dec_len, int):
                raise ValueError(
                    f"Valid maximum input length and maximum decoder length are required for static VMFB models, received ({max_inp_len}, {max_dec_len})"
                )
            return MoonshineStatic.from_vmfb(
                encoder,
                decoder,
                decoder_with_past,
                model_size,
                max_inp_len,
                max_dec_len,
            )
        return MoonshineStatic.from_onnx(
            encoder, decoder, decoder_with_past, model_size
        )
    else:
        if kind == "vmfb":
            return MoonshineDynamic(
                IREEInferenceRunner(encoder),
                IREEInferenceRunner(decoder_merged, function="merged"),
                model_size,
                max_inp_len
            )
        else:
            return MoonshineDynamic.from_onnx(encoder, decoder_merged, model_size, max_inp_len)


def _transcribe(wav: str | os.PathLike, runner: MoonshineDynamic | MoonshineStatic, tokenizer) -> str:
    data, _ = sf.read(wav, dtype="float32")
    speech = data.astype(np.float32)[np.newaxis, :]
    tokens = runner.run(speech)
    text = tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]
    return text


def main():
    runner = _load_moonshine(args.model_dir, args.model_size, args.max_inp_len, args.max_dec_len)
    tokenizer_file = hf_hub_download(f"UsefulSensors/moonshine-{args.model_size}", "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_file)
    try:
        for wav in args.inputs:
            transcribed = _transcribe(wav, runner, tokenizer)
            print(format_answer(transcribed, runner.last_infer_time, agent_name="Transcribed"))
    except KeyboardInterrupt:
        logger.info("Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moonshine Demo")
    parser.add_argument(
        "inputs",
        type=str,
        metavar="WAV",
        nargs="+",
        help="WAV files for inference",
    )
    parser.add_argument(
        "-m", "--model-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Path to Moonshine model directory",
    )
    parser.add_argument(
        "-s", "--model-size",
        type=str,
        required=True,
        choices=["base", "tiny"],
        help="Moonshine model size"
    )
    parser.add_argument(
        "--max-inp-len",
        type=int,
        help="Maximum input length (required for static models)",
    )
    parser.add_argument(
        "--max-dec-len",
        type=int,
        help="Maximum decoder length (required for static models)",
    )
    add_common_args(parser)
    add_logging_args(parser)
    args = parser.parse_args()

    configure_logging(args.logging)
    logger = logging.getLogger("Moonshine")
    logger.info("Starting demo...")

    main()
