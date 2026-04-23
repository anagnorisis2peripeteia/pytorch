import logging
from collections.abc import Callable

import torch
from torch._inductor.fx_passes.bucketing import (
    bucket_all_gather_by_mb,
    bucket_reduce_scatter_by_mb,
    BucketMode,
    is_all_gather_into_tensor as is_all_gather,
    is_reduce_scatter_tensor,
    is_wait_tensor,
    merge_all_gather,
    merge_reduce_scatter,
)
from torch.utils._ordered_set import OrderedSet


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_graph_input(node: torch.fx.Node) -> bool:
    return node.op == "placeholder"


def is_fsdp_all_gather(n):
    assert is_all_gather(n)
    while len(n.all_input_nodes) == 1:
        n = n.all_input_nodes[0]
        if n.op == "placeholder":
            return True
    return False


def is_fsdp_all_gather_wait(wait: torch.fx.Node) -> bool:
    # Assume all_gather_into_tensor input is either graph input
    # or dtype conversion of graph input
    ag_node = wait.args[0]  # type: ignore[arg-type, union-attr]
    return is_fsdp_all_gather(ag_node)


def is_graph_output(node: torch.fx.Node) -> bool:
    return all(user.op == "output" for user in node.users)


def is_fsdp_reduce_scatter_wait(wait: torch.fx.Node) -> bool:
    if is_graph_output(wait):
        return True

    if len(wait.users) == 1:
        user = next(iter(wait.users))
        assert user is not None
        return (
            is_graph_output(user)
            and user.op == "call_function"
            and user.target is torch.ops.prims.convert_element_type.default
        )

    return False


def _is_add(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is torch.ops.aten.add.Tensor


_LINEAR_REDUCE_OPS = OrderedSet(["sum", "avg"])


def _get_rs_args(node: torch.fx.Node) -> tuple | None:
    """Return (reduce_op, group_size, group_name) for a reduce_scatter_tensor node."""
    if not is_reduce_scatter_tensor(node):
        return None
    if node.args[1] not in _LINEAR_REDUCE_OPS:
        return None
    return (node.args[1], node.args[2], node.args[3])


def _collect_fusible_waits(
    node: torch.fx.Node,
    rs_args: tuple | None = None,
) -> tuple[list[torch.fx.Node], tuple] | None:
    """
    Walk an add-tree rooted at `node` and collect the wait_tensor nodes.
    Each wait_tensor must wait on a reduce_scatter, and all reduce_scatters
    must have identical (reduce_op, group_size, group_name) arguments.
    Returns None if the tree is not fully fusible.
    """
    if is_wait_tensor(node):
        if len(node.users) != 1:
            return None
        rs_node = node.args[0]
        assert isinstance(rs_node, torch.fx.Node)
        node_rs_args = _get_rs_args(rs_node)
        if node_rs_args is None or len(rs_node.users) != 1:
            return None
        if rs_args is not None and node_rs_args != rs_args:
            return None
        return [node], node_rs_args

    if _is_add(node):
        # pyrefly: ignore[bad-argument-type]
        left = _collect_fusible_waits(node.args[0], rs_args)
        if left is None:
            return None
        left_leaves, discovered_rs_args = left
        # pyrefly: ignore[bad-argument-type]
        right = _collect_fusible_waits(node.args[1], discovered_rs_args)
        if right is None:
            return None
        right_leaves, _ = right
        return left_leaves + right_leaves, discovered_rs_args

    return None


def dedup_fsdp_reduce_scatter(gm: torch.fx.GraphModule) -> None:
    """
    Fuse duplicate reduce_scatter ops whose waited results are summed.

    RS is linear, so RS(a) + RS(b) = RS(a + b). This pass rewrites
        rs_a = reduce_scatter(input_a, ...); wait_a = wait(rs_a)
        rs_b = reduce_scatter(input_b, ...); wait_b = wait(rs_b)
        result = add(wait_a, wait_b)
    into
        combined = add(input_a, input_b)
        rs = reduce_scatter(combined, ...)
        result = wait(rs)
    generalizing to N-way add trees.
    """
    graph = gm.graph
    changed = False

    for node in reversed(list(graph.nodes)):
        if node._erased:
            continue

        if not _is_add(node):
            continue

        result = _collect_fusible_waits(node)
        if result is None:
            continue
        wait_nodes, rs_args = result
        if len(wait_nodes) < 2:
            continue
        # wait_tensor and reduce_scatter_tensor are registered as
        # side-effectful ops, so eliminate_dead_code() won't remove them
        # by default. We override is_impure_node to allow removal of
        # only the specific nodes we've replaced with a fused version.
        replaced_nodes: OrderedSet[torch.fx.Node] = OrderedSet()
        for w in wait_nodes:
            replaced_nodes.add(w)
            # pyrefly: ignore[bad-argument-type]
            replaced_nodes.add(w.args[0])

        # pyrefly: ignore[missing-attribute]
        rs_inputs = [w.args[0].args[0] for w in wait_nodes]

        with graph.inserting_before(node):
            combined = rs_inputs[0]
            for inp in rs_inputs[1:]:
                combined = graph.call_function(
                    torch.ops.aten.add.Tensor, args=(combined, inp)
                )
                combined.meta["val"] = rs_inputs[0].meta.get("val")

            new_rs = graph.call_function(
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                args=(combined, *rs_args),
            )
            # pyrefly: ignore[missing-attribute]
            new_rs.meta["val"] = wait_nodes[0].args[0].meta.get("val")

            new_wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(new_rs,),
            )
            new_wait.meta["val"] = wait_nodes[0].meta.get("val")

        node.replace_all_uses_with(new_wait)
        graph.eliminate_dead_code(
            is_impure_node=lambda n: n.is_impure() and n not in replaced_nodes
        )
        changed = True

    if changed:
        graph.lint()
        gm.recompile()


def bucket_fsdp_all_gather(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode = "default",
) -> None:
    """
    Bucketing pass for SimpleFSDP all_gather ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float] | None): callback function that
            takes in bucket id and returns size of a bucket in megabytes.
    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    assert bucket_cap_mb_by_bucket_idx is not None
    ag_buckets = bucket_all_gather_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        filter_wait_node=is_fsdp_all_gather_wait,
    )
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets, mode)


def bucket_fsdp_reduce_scatter(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode = "default",
) -> None:
    """
    Bucketing pass for SimpleFSDP reduce_scatter ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float] | None): callback function that
            takes in bucket idx and returns size of a bucket in megabytes. By default
            torch._inductor.fx_passes.bucketing.bucket_cap_mb_by_bucket_idx_default is used.

    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    rs_buckets = bucket_reduce_scatter_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        filter_wait_node=is_fsdp_reduce_scatter_wait,
    )
    if len(rs_buckets) == 0:
        return
    merge_reduce_scatter(gm, rs_buckets, mode)
