# Owner(s): ["module: inductor"]
import unittest
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.async_compile import AsyncCompile, shutdown_compile_workers
from torch._inductor.compile_worker.subproc_pool import SubprocException
from torch._inductor.runtime.triton_compat import Config
from torch._inductor.runtime.triton_heuristics import (
    generate_lookup_hash_from_source_code,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    requires_gpu,
    requires_triton,
)


try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False


CUTEDSL_ADD_TEMPLATE = r"""
{{gen_defines()}}

@cute.kernel
def {{kernel_name}}_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx
    m, n = gA.shape

    if thread_idx < m * n:
        mi = thread_idx // n
        ni = thread_idx % n

        if mi < m and ni < n:
            gC[mi, ni] = gA[mi, ni] + gB[mi, ni]

@cute.jit
def {{kernel_name}}_jit(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream):
    {{gen_defines()}}
    m, n = mA.shape
    total_threads = m * n
    num_blocks = (total_threads + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    kernel = {{kernel_name}}_kernel(mA, mB, mC)
    kernel.launch(
        grid=[num_blocks, 1, 1],
        block=[THREADS_PER_BLOCK, 1, 1],
        stream=stream
    )

{{def_kernel("input_a", "input_b")}}
    cute_a = from_dlpack(input_a)
    cute_b = from_dlpack(input_b)
    cute_c = from_dlpack({{get_output()}})

    {{kernel_name}}_jit(cute_a, cute_b, cute_c, cuda.CUstream(stream))
    return {{get_output()}}
"""


@instantiate_parametrized_tests
class TestAsyncCompile(TestCase):
    @requires_gpu()
    @requires_triton()
    @parametrize("method", ("subprocess", "fork", "spawn"))
    def test_pool(self, method):
        def fn(x, y):
            return x + y

        x = torch.rand(10).to(GPU_TYPE)
        y = torch.rand(10).to(GPU_TYPE)

        with config.patch("worker_start_method", method):
            shutdown_compile_workers()
            AsyncCompile.wait_pool_ready()

            with fresh_cache():
                compiled_fn = torch.compile(fn)
                self.assertEqual(fn(x, y), compiled_fn(x, y))

    @requires_gpu()
    @requires_triton()
    def test_bad_kernel(self):
        shutdown_compile_workers()

        with config.patch(worker_start_method="subprocess", compile_threads=8):
            async_compile = AsyncCompile()
            AsyncCompile.wait_pool_ready()
            with self.assertRaises(SubprocException):
                async_compile.triton(
                    "fake_kernel_name", source_code="This definitely doesn't exist"
                ).result()

    @requires_gpu()
    @requires_triton()
    def test_wait_pool_ready(self):
        shutdown_compile_workers()

        with config.patch(worker_start_method="subprocess", compile_threads=8):
            AsyncCompile.wait_pool_ready()
            self.assertTrue(AsyncCompile._ready_future.done())
            self.assertTrue(AsyncCompile.use_process_pool())

    @requires_gpu()
    @requires_triton()
    @patch("torch._inductor.runtime.coordinate_descent_tuner.CoordescTuner.autotune")
    @parametrize("method", ("subprocess", "fork", "spawn"))
    def test_autotune_lookup_table(self, mock_autotune, method):
        def f(a, b):
            return (a @ b).to(torch.float32).sum(dim=1)

        # Fake name to make sure the lookup table is name agnostic
        # When codegen/triton.py is changed, func_def must be updated
        loop_header = (
            "for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):"
            if torch.version.hip
            else "for r0_offset in tl.range(0, r0_numel, R0_BLOCK):"
        )

        func_def = f"""
def triton_fused_fake_name(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 11776
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    {loop_header}
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 11776*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)

"""

        fn_hash = generate_lookup_hash_from_source_code(
            str({"x": 1024, "r0_": 16384}), func_def
        )
        block_configs = {
            "XBLOCK": 1,
            "R0_BLOCK": 128,
        }
        num_warps = 16
        num_stages = 1
        autotune_lookup_table = {
            fn_hash: {**block_configs, "num_warps": num_warps, "num_stages": num_stages}
        }
        autotune_config = Config(
            block_configs, num_warps=num_warps, num_stages=num_stages
        )
        mock_autotune.return_value = autotune_config

        a = torch.randn(1152, 1024, device=GPU_TYPE, dtype=torch.float16).T
        b = torch.randn(1152, 11776, device=GPU_TYPE, dtype=torch.float16)
        compiled_f = torch.compile(f)

        with config.patch(
            {
                "autotune_lookup_table": autotune_lookup_table,
                "coordinate_descent_tuning": True,
                "worker_start_method": method,
            }
        ):
            shutdown_compile_workers()
            AsyncCompile.wait_pool_ready()
            with fresh_cache():
                compiled_f(a, b)

        # Check that the input to coordinate descent (the resulting chosen config)
        # is the same as the one in the lookup table
        mock_autotune.assert_called_once()
        args, _ = mock_autotune.call_args
        self.assertTrue(isinstance(args[1], Config))

        self.assertEqual(args[1].kwargs, autotune_config.kwargs)
        self.assertEqual(args[1].num_warps, autotune_config.num_warps)
        self.assertEqual(args[1].num_stages, autotune_config.num_stages)


@unittest.skipUnless(HAS_CUTLASS, "requires cutlass")
@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCuteDSLSubprocessCompile(TestCase):
    def _compile_and_run_add(self, template_name):
        """Compile a CuteDSL add kernel via torch.compile and verify correctness."""
        from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate
        from torch._inductor.ir import TensorBox
        from torch._inductor.lowering import lowerings
        from torch._inductor.utils import run_and_get_code

        template = CuteDSLTemplate(
            name=template_name,
            source=CUTEDSL_ADD_TEMPLATE,
        )

        def cutedsl_add_lowering(a: TensorBox, b: TensorBox) -> TensorBox:
            choices = []
            error = template.maybe_append_choice(
                choices,
                input_nodes=[a, b],
                layout=a.get_layout(),
                THREADS_PER_BLOCK=256,
            )
            if error or not choices:
                default_lowering = lowerings[torch.ops.aten.add.Tensor]
                return default_lowering(a, b)
            return choices[0].output_node()

        with patch.dict(lowerings, {torch.ops.aten.add.Tensor: cutedsl_add_lowering}):

            def test_add(x, y):
                return x + y

            x = torch.randn(128, 4, device="cuda", dtype=torch.float32)
            y = torch.randn(128, 4, device="cuda", dtype=torch.float32)

            compiled_fn = torch.compile(test_add, backend="inductor")
            result, (code,) = run_and_get_code(compiled_fn, x, y)

            self.assertIn("cute", code.lower())
            expected = x + y
            self.assertEqual(result, expected)

    def test_cutedsl_subprocess_e2e(self):
        shutdown_compile_workers()
        with config.patch(worker_start_method="subprocess", compile_threads=4):
            AsyncCompile.wait_pool_ready()
            self.assertTrue(AsyncCompile.use_process_pool())
            with (
                fresh_cache(),
                patch.object(
                    AsyncCompile,
                    "_load_kernel_wrapper",
                    autospec=True,
                    side_effect=AsyncCompile._load_kernel_wrapper,
                ) as mock_reload,
            ):
                self._compile_and_run_add("test_add_subprocess")
                mock_reload.assert_called()

    def test_cutedsl_synchronous_e2e(self):
        with config.patch(compile_threads=1):
            with (
                fresh_cache(),
                patch.object(
                    AsyncCompile,
                    "_load_kernel_wrapper",
                    autospec=True,
                    side_effect=AsyncCompile._load_kernel_wrapper,
                ) as mock_reload,
            ):
                self._compile_and_run_add("test_add_synchronous")
                mock_reload.assert_not_called()

    def test_cutedsl_bad_source_subprocess(self):
        shutdown_compile_workers()
        with config.patch(worker_start_method="subprocess", compile_threads=4):
            AsyncCompile.wait_pool_ready()
            self.assertTrue(AsyncCompile.use_process_pool())
            async_compile = AsyncCompile()

            with self.assertRaises(SubprocException):
                async_compile.cutedsl(
                    "bad_kernel", "this is not valid python!!!"
                ).result()

    def test_cutedsl_missing_entry_point_subprocess(self):
        shutdown_compile_workers()
        with config.patch(worker_start_method="subprocess", compile_threads=4):
            AsyncCompile.wait_pool_ready()
            self.assertTrue(AsyncCompile.use_process_pool())
            async_compile = AsyncCompile()

            with self.assertRaises(SubprocException):
                async_compile.cutedsl(
                    "test_kernel", "import torch\ndef other_func(): pass\n"
                ).result()


if __name__ == "__main__":
    run_tests()
