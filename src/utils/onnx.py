import argparse
import os
from collections import defaultdict

import onnx
import onnx_graphsurgeon as gs


def add_onnx_args(
    parser: argparse.ArgumentParser,
    *,
    model_dtypes: list[str] | None = None,
    allow_no_opt: bool = True,
):
    if model_dtypes:
        parser.add_argument(
            "-d", "--dtype",
            type=str,
            metavar="DTYPE",
            choices=model_dtypes,
            default=model_dtypes[0],
            help="Model data type (default: %(default)s, choices: %(choices)s)"
        )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default="models/onnx",
        metavar="DIR",
        help="Directory for saving ONNX models (default: %(default)s)",
    )
    parser.add_argument(
        "--show-model-info",
        action="store_true",
        default=False,
        help="Show ONNX model inputs/outputs and ops information",
    )
    if allow_no_opt:
        parser.add_argument(
            "--no-optimize",
            action="store_true",
            default=False,
            help="Do no optimize exported ONNX models via onnxruntime",
        )


def print_onnx_model_inputs_outputs_info(model: onnx.ModelProto | str | os.PathLike):
    if isinstance(model, (str, os.PathLike)):
        model = onnx.load(model)

    model_gs = gs.import_onnx(model)

    input_consumers = defaultdict(list)
    graph_input_names = {i.name: (i.shape, i.dtype) for i in model_gs.inputs}

    for node in model_gs.nodes:
        for input in node.inputs:
            name = input.name
            if name in graph_input_names:
                input_consumers[name].append(node)

    print(f"\n\nModel inputs info:\n")
    for name in sorted(graph_input_names):
        shape, dtype = graph_input_names[name]
        consumers = input_consumers.get(name, [])
        if consumers:
            consumers = "\n\t".join([f"'{node.name}'" for node in consumers])
            print(f"Input '{name}' ({dtype}{shape}) consumed by:\n\t{consumers}")
        else:
            print(f"Input '{name}' ({dtype}{shape}) is not consumed by any node")

    output_names = {o.name: (o.shape, o.dtype) for o in model_gs.outputs}
    output_to_node = {out.name: node for node in model_gs.nodes for out in node.outputs}

    print(f"\n\nModel outputs info:\n")
    for name, (shape, dtype) in output_names.items():
        node = output_to_node.get(name)
        if node:
            print(f"Output '{name}' ({dtype}{shape}) produced by:\n\t'{node.name}'")
        elif name in {i.name for i in model_gs.graph.input}:
            print(f"Output '{name}' is a passthrough from graph input")
        elif name in {init.name for init in model_gs.graph.initializer}:
            print(f"Output '{name}' is from initializer")
        else:
            print(f"Output '{name}' has no known producer (invalid?)")


def get_model_ops_count(model: onnx.ModelProto) -> dict[str, int]:
    op_counts = {}
    for node in model.graph.node:
        if op_counts.get(node.op_type) is None:
            op_counts[node.op_type] = 0
        op_counts[node.op_type] += 1

    op_counts = dict(sorted(op_counts.items(), key=lambda item: item[1], reverse=True))
    return op_counts


def check_dynamic_shapes(model: onnx.ModelProto) -> dict[str, list[int | str]]:

    def _is_static_shape(shape: list[int | str] | None) -> bool:
        return shape is not None and all(isinstance(d, int) and d > 0 for d in shape)

    dynamic_shapes: dict[str, list[int | str]] = {}
    graph = gs.import_onnx(model)
    for tensor in graph.inputs + graph.outputs:
        if not _is_static_shape(tensor.shape):
            print(
                f"Static model check failed: I/O tensor '{tensor.name}' has non-static shape {tensor.shape}"
            )
            dynamic_shapes[tensor.name] = tensor.shape
    for tensor_name, tensor in graph.tensors().items():
        if not _is_static_shape(tensor.shape):
            print(
                f"Static model check failed: Graph tensor '{tensor_name}' has non-static shape {tensor.shape}"
            )
            dynamic_shapes[tensor_name] = tensor.shape
    return dynamic_shapes


if __name__ == "__main__":
    pass
