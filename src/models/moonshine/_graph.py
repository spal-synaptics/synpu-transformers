import logging
from functools import wraps, WRAPPER_ASSIGNMENTS
from typing import ParamSpec, Callable, Concatenate

import onnx
import onnx_graphsurgeon as gs
import numpy as np


logger = logging.getLogger("MoonshineGraphEditor")
P = ParamSpec("P")
ASSIGNED = tuple(x for x in WRAPPER_ASSIGNMENTS if x != "__annotations__")

def graph_edit(edit_fn: Callable[Concatenate[gs.Graph, gs.Node, P], object]):
    @wraps(edit_fn, assigned=ASSIGNED)
    def run(component: str, graph: gs.Graph, **kwargs: P.kwargs) -> gs.Graph:
        for node in list(graph.nodes):
            edit_fn(component, graph, node, **kwargs)
        return graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=True
        ).toposort()
    return run


@graph_edit
def remove_isNaN(
    component: str,
    graph: gs.Graph,
    node: gs.Node
):
    if node.op == "IsNaN":
        graph.remove_unsupported_isNaN(node)
        logger.info("[%s] Removed unsupported IsNaN op '%s'", component, node.name)


@graph_edit
def move_output_from_concat(
    component: str,
    graph: gs.Graph,
    node: gs.Node,
    *,
    output_names: list[str],
    pad_len: str
):
    if node.op == "Concat" and node.outputs[0].name in output_names:
        output_name = node.outputs[0].name
        consumers: list[gs.Node] = list(node.outputs[0].outputs)
        for consumer in consumers:
            if consumer.op == "Pad":
                graph.move_output_from_concat_to_slice(node, consumer, pad_len)
                logger.info("[%s] Moved output '%s' to Pad node '%s'", component, output_name, consumer.name)


@graph_edit
def replace_dynamic_kv_cache(
    component: str,
    graph: gs.Graph,
    node: gs.Node,
    *,
    output_names: list[str],
    cur_len: gs.Variable,
    max_tokens: int
):
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
        graph.replace_kv_concat_with_mask(node, cur_len, max_tokens)
        logger.info("[%s] Added static KV cache for output '%s'", component, cache_output)


@graph_edit
def mask_future_attn_scores(
    component: str,
    graph: gs.Graph,
    node: gs.Node,
    *,
    cur_len: gs.Variable,
    max_tokens: int
):
    if node.op == "Softmax" and node.name.endswith("self_attn/Softmax"):
        if (prod := node.i()).op != "Add":
            raise ValueError(
                f"Causal Attention Mask: '{node.name}' expected producer to be Add node, got {prod.op} ({prod.name})"
            )
        graph.add_causal_attn_score_mask(prod, cur_len, max_tokens)
        logger.info("[%s] Added causal attention mask to scores at node '%s'", component, node.name)


@graph_edit
def add_curr_len_input(
    component: str,
    graph: gs.Graph,
    node: gs.Node,
    *,
    cur_len: gs.Variable
):
    if node.op == "Shape" and "past_key_values" in node.inputs[0].name:
        graph.replace_dynamic_seq_len_getter(node, cur_len)
        logger.info("[%s] Replaced dynamic seq len getter at node '%s'", component, node.name)


@graph_edit
def convert_to_static_index(
    component: str,
    graph: gs.Graph,
    node: gs.Node
):
    if (
        node.op == "Range"
        and node.i(1).op == "Add"
        and any(inp is node.inputs[0] for inp in node.i(1).inputs)
    ):
        graph.replace_dynamic_range_index(node)
        logger.info("[%s] Replaced dynamic range index for node '%s'", component, node.name)


@gs.Graph.register()
def remove_unsupported_isNaN(self: gs.Graph, is_nan_node: gs.Node) -> None:
    """
    Remove unsupported `IsNaN -> Where` operation.

    Args:
        self (gs.Graph): ONNX graph
        is_nan_op (gs.Node): IsNaN node to replace

    Raises:
        ValueError: If `is_nan_op` is not a `IsNaN` op
        ValueError: If `is_nan_op` is not consumed by a `Where` op
    """
    if is_nan_node.op != "IsNaN":
        raise ValueError(
            f"Expected IsNaN node, got {is_nan_node.op} for IsNaN replacement"
        )
    producer: gs.Tensor = is_nan_node.inputs[0]
    where_node: gs.Node = is_nan_node.o()
    if where_node.op != "Where":
        raise ValueError(
            f"Expected Where node consumer, got {where_node.op} for IsNaN replacement"
        )
    where_out: gs.Variable = where_node.outputs[0]
    consumers: list[gs.Node] = list(where_out.outputs)
    for consumer in consumers:
        for i, inp in enumerate(consumer.inputs):
            if inp is where_out:
                consumer.inputs[i] = producer

    # disconnect IsNaN -> Where chain
    is_nan_node.inputs.clear()
    where_node.inputs.clear()
    where_node.outputs.clear()


@gs.Graph.register()
def move_output_from_concat_to_slice(
    self: gs.Graph, concat_node: gs.Node, pad_node: gs.Node, pad_len: int
) -> None:
    """
    Move model output from Concat node to consumer Pad node.

    This is requried to prevent errors with Acuity compilation.

    Args:
        self (gs.Graph): ONNX graph
        concat_node (gs.Node): Concat node original model output
        pad_node (gs.Node): Pad node to move model output
        cur_len (gs.Variable): Current sequence length input
        padding (int): Padding length to drop

    Raises:
        ValueError: If the `concat_node` is not a `Concat` op
        ValueError: If the `pad_node` is not a `Pad` op
    """
    if concat_node.op != "Concat":
        raise ValueError(
            f"Expected Concat node, got {concat_node.op} for moving output from Concat to Slice"
        )
    if pad_node.op != "Pad":
        raise ValueError(
            f"Expected Pad node, got {pad_node.op} for moving output from Concat to Slice"
        )
    concat_output: gs.Variable = concat_node.outputs[0]
    pad_output: gs.Variable = pad_node.outputs[0]

    tensors = self.tensors()
    if not (output_slice_starts := tensors.get("output_slice_starts")):
        output_slice_starts = gs.Constant(
            "output_slice_starts", np.array([0], dtype=np.int64)
        )
    if not (output_slice_ends := tensors.get("output_slice_ends")):
        output_slice_ends = gs.Constant(
            "output_slice_ends", np.array([-pad_len], dtype=np.int64)
        )
    if not (output_slice_axes := tensors.get("output_slice_axes")):
        output_slice_axes = gs.Constant(
            "output_slice_axes", np.array([3], dtype=np.int64)
        )
    if not (output_slice_steps := tensors.get("output_slice_steps")):
        output_slice_steps = gs.Constant(
            "output_slice_steps", np.array([1], dtype=np.int64)
        )
    slice_output: gs.Variable = self.layer(
        name=pad_output.name + "_slice",
        op="Slice",
        inputs=[
            pad_output,
            output_slice_starts,
            output_slice_ends,
            output_slice_axes,
            output_slice_steps,
        ],
        outputs=[
            gs.Variable(
                concat_output.name, dtype=concat_output.dtype, shape=concat_output.shape
            )
        ],
    )[0]

    for i, output in enumerate(self.outputs):
        if output is concat_output:
            self.outputs[i] = slice_output

    orig = concat_output.name
    concat_output.name = orig + "_prepad"
    slice_output.name = orig


@gs.Graph.register()
def replace_kv_concat_with_mask(
    self: gs.Graph, concat_node: gs.Node, cur_len: gs.Variable, L: int
) -> None:
    """
    Replace dynamic Concat update of the KV cache with a static in-place blend.

    `cache[i] = new_value if i == cur_len else cache[i]`

    Args:
        self (gs.Graph): ONNX graph
        concat_node (gs.Node): Concat node to replace
        cur_len (gs.Variable): Current sequence length input
        L (int): Maximum sequence length

    Raises:
        ValueError: If the `concat_node` is not a `Concat` op

    Notes:
        - Builds a mask that is true for the current position
        - Blends the new cache value into the existing cache using the mask
        - Disconnects old Concat node from the graph
        - Optimizers may CSE-deduplicate identical masks into one shared tensor
    """
    if concat_node.op != "Concat":
        raise ValueError(
            f"Expected Concat node, got {concat_node.op} for KV cache update"
        )
    past_cache_vals, new_cache_val = concat_node.inputs
    output = concat_node.outputs[0]

    # create mask for current position
    mask_shape = [1, 1, L, 1]
    if not (time_ids := self.tensors().get("time_ids")):
        time_ids = gs.Constant(
            "time_ids", np.arange(L, dtype=np.int64).reshape(*mask_shape)
        )
    mask = self.layer(
        name=output.name + "_update_mask",
        op="Equal",
        inputs=[time_ids, cur_len],
        outputs=[
            gs.Variable(
                f"{output.name}_mask_eq", dtype=onnx.TensorProto.BOOL, shape=mask_shape
            )
        ],
    )[0]

    # blend new value into cache using the mask
    self.layer(
        name=output.name + "_blend_kv",
        op="Where",
        inputs=[mask, new_cache_val, past_cache_vals],
        outputs=[output],
    )

    # disconnect Concat node
    concat_node.inputs.clear()
    concat_node.outputs.clear()


@gs.Graph.register()
def add_causal_attn_score_mask(
    self: gs.Graph,
    add_node: gs.Node,
    cur_len: gs.Variable,
    L: int,
) -> None:
    """
    Add a causal mask to attention scores to block future positions.

    Enforces left-to-right causality by assigning a large negative value to positions > `cur_len`.

    Args:
        self (gs.Graph): ONNX graph
        add_node (gs.Node): Add node that produces attention scores
        cur_len (gs.Variable): Current sequence length input
        L (int): Maximum sequence length

    Raises:
        ValueError: If `add_node` is not an `Add` op

    Notes:
        - Creates a mask that is only true for positions <= cur_len
        - Rewires the attention score producer to use this mask
        - Optimizers may CSE-deduplicate identical masks into one shared tensor
    """
    if add_node.op != "Add":
        raise ValueError(
            f"Expected Add node, got {add_node.op} for attention score producer"
        )

    # create bool mask where positions > cur_len are effectively blocked
    # by being set to a large negative value
    mask_shape = [1, 1, 1, L]
    if not (time_axis := self.tensors().get("time_axis")):
        time_axis = gs.Constant(
            "time_axis", np.arange(L, dtype=np.int64).reshape(*mask_shape)
        )
    if not (attn_mask_keep := self.tensors().get("attn_mask_keep")):
        attn_mask_keep = gs.Constant(
            "attn_mask_keep", np.asarray(0.0, dtype=np.float32)
        )
    if not (attn_mask_block := self.tensors().get("attn_mask_block")):
        attn_mask_block = gs.Constant(
            "attn_mask_block", np.asarray(-1e9, dtype=np.float32)
        )
    mask_lte = self.layer(
        name=add_node.name + "_lte_cur_len",
        op="LessOrEqual",
        inputs=[time_axis, cur_len],
        outputs=[
            gs.Variable(
                add_node.name + "_less", dtype=onnx.TensorProto.BOOL, shape=mask_shape
            )
        ],
    )[0]
    mask = self.layer(
        name=add_node.name + "_mask_attn",
        op="Where",
        inputs=[mask_lte, attn_mask_keep, attn_mask_block],
        outputs=[
            gs.Variable(add_node.name + "_where", dtype=np.float32, shape=mask_shape)
        ],
    )[0]

    # rewire Add node to use mask
    add_node.inputs[1] = mask


@gs.Graph.register()
def replace_dynamic_seq_len_getter(
    self: gs.Graph, shape_node: gs.Node, cur_len: gs.Variable
):
    """
    Replace dynamic Shape->Gather path with a static sequence length input.

    Removes the runtime-calculated sequence length and replaces it with
    the model input `cur_len`.

    Args:
        self (gs.Graph): ONNX graph
        shape_node (gs.Node): Shape node to replace
        cur_len (gs.Variable): Current sequence length input

    Raises:
        ValueError: If `shape_node` is not a `Shape` op
        ValueError: If the Gather node after Shape is not a `Gather` op

    Notes:
        - Replaces `Shape(past_key_values) -> Gather(i=2)` with `cur_len`
        - Disconnects original Shape and Gather nodes
    """
    if shape_node.op != "Shape":
        raise ValueError(
            f"Expected Shape node, got {shape_node.op} for dynamic shape replacement"
        )
    gather_node: gs.Node = shape_node.o()
    if not isinstance(gather_node, gs.Node) and gather_node.op == "Gather":
        raise ValueError(f"Expected Gather node after Shape, got {gather_node}")

    gather_out: gs.Variable = gather_node.outputs[0]
    consumers: list[gs.Node] = list(gather_out.outputs)
    for consumer in consumers:
        for i, inp in enumerate(consumer.inputs):
            if inp is gather_out:
                consumer.inputs[i] = cur_len

    # disconnect Shape + Gather branch
    shape_node.inputs.clear()
    gather_node.outputs.clear()


@gs.Graph.register()
def replace_dynamic_range_index(self: gs.Graph, range_node: gs.Node):
    """
    Replaces redundant index computation `Range(start, start + 1, 1)` by wiring consumers to directly accept `start`.

    Args:
        self (gs.Graph): ONNX graph
        range_node (gs.Node): Range node to replace

    Raises:
        ValueError: If `range_node` is not a `Range` op
        ValueError: If `limit` is not produced by an `Add` op
        ValueError: If `start` and `limit` don't share a common producer

    Notes:
      - Directly connects `start` to consumers of `range_node`
      - Disconnects `range_node` from the graph
    """
    if range_node.op != "Range":
        raise ValueError(
            f"Expected Range node, got {range_node.op} for dynamic range replacement"
        )
    start = range_node.inputs[0]
    limit_prod = range_node.i(1)
    if limit_prod.op != "Add":
        raise ValueError(
            f"Expected Add node for limit, got {limit_prod.op} for dynamic range replacement"
        )
    if not any(inp is start for inp in limit_prod.inputs):
        raise ValueError(
            f"Range node and limit node must have common producer for dynamic range replacement"
        )
    range_out: gs.Variable = range_node.outputs[0]
    consumers: list[gs.Node] = list(range_out.outputs)
    for consumer in consumers:
        for i, inp in enumerate(consumer.inputs):
            if inp is range_out:
                consumer.inputs[i] = start

    # disconnect Range node
    range_node.inputs.clear()
    range_node.outputs.clear()
