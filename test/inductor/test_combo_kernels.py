# Owner(s): ["module: inductor"]

import contextlib
import json
import logging
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch._inductor
from torch._inductor.memory import BufferInfo
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import fresh_cache, run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
    skipIfXpu,
    TestCase,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU_AND_TRITON
from torch.testing._internal.triton_utils import (
    requires_cuda_and_triton,
    requires_gpu_and_triton,
    requires_xpu_and_triton,
)


aten = torch.ops.aten

try:
    try:
        from .test_torchinductor import check_model, check_model_gpu
    except ImportError:
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            check_model,
            check_model_gpu,
        )
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise


@instantiate_parametrized_tests
class ComboKernelTests(TestCase):
    check_model_gpu = check_model_gpu
    check_model_cpu = check_model
    check_kernel_count = True
    combo_kernel_per_subkernel_blocks = False

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                    "combo_kernel_per_subkernel_blocks": self.combo_kernel_per_subkernel_blocks,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    def test_activation_functions(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    def test_reduce_functions(self):
        def test_reduce(a, b, c, d):
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(torch._inductor.metrics.generated_kernel_count <= 2)

    @requires_gpu_and_triton
    def test_mutated_args(self):
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    def test_reduce_split(self):
        def fn(a, b):
            a1 = torch.linalg.vector_norm(a)
            b1 = torch.sum(b, dim=0)
            return a1, b1

        inps = [
            torch.rand(2048, 512, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)

    @requires_gpu_and_triton
    def test_2d_blocking_partitioning(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_gpu(
            fn,
            (
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(40, 30, device=GPU_TYPE),
                torch.rand(36, 40, device=GPU_TYPE),
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(30, 40, device=GPU_TYPE).t(),
                torch.rand(40, 36, device=GPU_TYPE).t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_gpu_and_triton
    def test_persistent_reduction_size_hint(self):
        def fn(x, y):
            return x.max(1), y.min(1)

        inps = (
            torch.rand(768, 16, device=GPU_TYPE),
            torch.rand(768, 32, device=GPU_TYPE),
        )

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        FileCheck().check("triton_heuristics.persistent_reduction").check(
            "size_hints={'x': 1024, 'r0_': 32}"
        ).run(code[0])
        self.assertEqual(out_eager, out_compiled)

    @requires_gpu_and_triton
    def test_fuse_mix_order_reductions_combo_kernels(self):
        def fn(x, y, z):
            # FusedMixOrderReductions produces row_sum (buf0)
            row_sum = x.sum(dim=1)
            col_sum = x.sum(dim=0)

            # consumer of row_sum - excluded from combo kernels
            row_sum_reduced = row_sum.sum()  # reads buf0

            # independent reductions - combo-kerneled
            y_sum = y.sum()
            z_sum = z.sum()

            return row_sum_reduced, col_sum, y_sum, z_sum

        inps = [
            torch.rand(8192, 1024, device=GPU_TYPE),
            torch.rand(2048, device=GPU_TYPE),
            torch.rand(2048, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        self.assertEqual(out_eager, out_compiled)
        # [row_sum, col_sum] will became 1 kernel MixOrderReductionGrid
        # [row_sum_reduced] will become a separate kernel due to the consumer
        # [y_sum, z_sum] will become a combo kernel
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 3)

    @requires_gpu_and_triton
    def test_combo_kernel_scalar_store_broadcast(self):
        def fn(a, b, c, d):
            scalar_sum = a + b
            vector_result = c.sum(dim=1)
            scalar_red = (d * scalar_sum).sum()
            return scalar_sum, vector_result, scalar_red

        inps = [
            torch.tensor(1.0, device=GPU_TYPE),
            torch.tensor(2.0, device=GPU_TYPE),
            torch.randn(2048, 2048, device=GPU_TYPE),
            torch.randn(2048, 1, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        torch.testing.assert_close(out_eager, out_compiled, rtol=1e-4, atol=1e-4)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    @parametrize("benchmark_combo_kernel", [False, True])
    def test_single_combo_kernels(self, benchmark_combo_kernel):
        def fn(a, b):
            c = torch.add(a, 1)
            d = torch.add(b, 1)
            return c, d

        inps = [
            torch.rand(8192, 8192, device=GPU_TYPE),
            torch.rand(100, 100, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        with torch._inductor.config.patch(
            "benchmark_combo_kernel", benchmark_combo_kernel
        ):
            fn_c = torch.compile(fn)
            out_compiled, code = run_and_get_code(fn_c, *inps)
            self.assertEqual(out_eager, out_compiled)
            # With benchmark_combo_kernel=True, kernel count is 2x due to benchmarking
            # (1x benchmarking + 1x actual codegen, single-node code generation skipped)
            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count,
                4 if benchmark_combo_kernel else 2,
            )
            # Verify kernels are regular pointwise, not combo kernels
            FileCheck().check("triton_heuristics.pointwise").check(
                "'grid_type': 'Grid1D'"
            ).check_not("combo_grid_meta").run(code[0])

    @requires_gpu_and_triton
    def test_combo_kernel_per_config_subkernel_poi(self):
        def fn(a, b):
            o1 = a * 2.0
            o2 = b + 1.0
            return o1, o2

        inps = [
            torch.randn(512, device=GPU_TYPE),
            torch.randn(524288, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
        if torch._inductor.config.combo_kernel_per_subkernel_blocks:
            FileCheck().check("XBLOCK_0 : tl.constexpr").check(
                "XBLOCK_1 : tl.constexpr"
            ).run(code[0])
        else:
            FileCheck().check_not("XBLOCK_0 : tl.constexpr").run(code[0])

    @requires_gpu_and_triton
    def test_combo_kernel_per_config_subkernel_per(self):
        def fn(a, b):
            return a.sum(dim=-1), b.sum(dim=-1)

        inps = [
            torch.randn(1024, 64, device=GPU_TYPE),
            torch.randn(1024, 512, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        torch.testing.assert_close(out_eager, out_compiled, atol=1e-4, rtol=1e-4)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
        if torch._inductor.config.combo_kernel_per_subkernel_blocks:
            # Per-subkernel: signature has XBLOCK_0, XBLOCK_1
            FileCheck().check("R0_BLOCK_0: tl.constexpr = 64").check(
                "R0_BLOCK_1: tl.constexpr = 512"
            ).run(code[0])
        else:
            FileCheck().check("XBLOCK : tl.constexpr").check_not(
                "XBLOCK_0 : tl.constexpr"
            ).run(code[0])

    @requires_gpu_and_triton
    def test_combo_kernel_per_config_subkernel_red_per(self):
        def fn(a, b):
            r1 = a.sum(dim=-1)
            r2 = b.sum(dim=-1)
            return r1, r2

        inps = [
            torch.randn(512, 128, device=GPU_TYPE),  # Persistent (r0=128)
            torch.randn(256, 2048, device=GPU_TYPE),  # Regular (r0=2048)
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        torch.testing.assert_close(out_eager, out_compiled, atol=1e-4, rtol=1e-4)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
        if torch._inductor.config.combo_kernel_per_subkernel_blocks:
            FileCheck().check("XBLOCK_0 : tl.constexpr").check(
                "XBLOCK_1 : tl.constexpr, R0_BLOCK_1 : tl.constexpr"
            ).run(code[0])
        else:
            FileCheck().check_not("XBLOCK_0 : tl.constexpr").run(code[0])

    @requires_gpu_and_triton
    def test_combo_kernel_per_config_subkernel_red(self):
        def fn(a, b):
            r1 = a.sum(dim=(0, 2))
            r2 = b.sum(dim=(0, 2))
            return r1, r2

        inps = [
            torch.randn(32, 64, 128, device=GPU_TYPE),
            torch.randn(32, 64, 128, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        torch.testing.assert_close(out_eager, out_compiled, atol=1e-4, rtol=1e-4)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
        if torch._inductor.config.combo_kernel_per_subkernel_blocks:
            FileCheck().check(
                " XBLOCK_0 : tl.constexpr, R0_BLOCK_0 : tl.constexpr"
            ).check("XBLOCK_1 : tl.constexpr, R0_BLOCK_1 : tl.constexpr").run(code[0])
        else:
            FileCheck().check("XBLOCK : tl.constexpr").check(
                "R0_BLOCK : tl.constexpr"
            ).check_not("XBLOCK_0 : tl.constexpr").run(code[0])

    @skipIfRocm
    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        {
            "triton.prefer_nd_tiling": True,
            "triton.tile_reductions": True,
        }
    )
    def test_combo_kernel_per_config_subkernel_r0_r1(self):
        def fn(a, b):
            return a.sum(dim=(1, 2)), b.sum(dim=(1, 2))

        inps = [
            torch.randn(32, 16, 64, device=GPU_TYPE).permute(1, 0, 2),
            torch.randn(32, 16, 64, device=GPU_TYPE).permute(1, 0, 2),
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        self.assertEqual(out_eager, out_compiled)
        # 2D reduction kernels (r0_, r1_) are separated from combo kernels
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        {
            "triton.prefer_nd_tiling": True,
            "triton.max_tiles": 3,
        }
    )
    def test_combo_kernel_per_config_subkernel_poi_3d(self):
        def fn(a, b):
            return a + 1.0, b * 2.0

        inps = [
            torch.randn(16, 16, 16, device=GPU_TYPE)[:8, :8, :8],
            torch.randn(16, 16, 16, device=GPU_TYPE)[:8, :8, :8],
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        self.assertEqual(out_eager, out_compiled)
        # 3D poi (x, y, z) are separated from combo kernels
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @skipIfXpu(msg="Profiler JSON traceEvents is not supported on XPU")
    @requires_gpu_and_triton
    def test_combo_kernel_per_config_subkernel_block_size(self):
        from torch.profiler import ProfilerActivity

        def fn(t0, t1, t2, t3, t4, t5, t6, t7):
            o0 = t0.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=12,    xnumel=50176
            o1 = t1.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=192,   xnumel=9
            o2 = t2.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=4096,  xnumel=9
            o3 = t3.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=8192,  xnumel=9
            o4 = t4.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=16384, xnumel=9
            o5 = t5.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=32768, xnumel=9
            o6 = t6.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=65536, xnumel=9
            o7 = t7.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=65536, xnumel=9
            return o0, o1, o2, o3, o4, o5, o6, o7

        inps = [
            torch.randn(4, 3, 224, 224, device=GPU_TYPE),
            torch.randn(64, 3, 3, 3, device=GPU_TYPE),
            torch.randn(64, 64, 3, 3, device=GPU_TYPE),
            torch.randn(128, 64, 3, 3, device=GPU_TYPE),
            torch.randn(128, 128, 3, 3, device=GPU_TYPE),
            torch.randn(256, 128, 3, 3, device=GPU_TYPE),
            torch.randn(256, 256, 3, 3, device=GPU_TYPE),
            torch.randn(256, 256, 3, 3, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)

        with tempfile.NamedTemporaryFile(suffix=".json") as trace_file:
            trace_path = trace_file.name
            activity = getattr(ProfilerActivity, GPU_TYPE.upper())

            with torch.profiler.profile(
                activities=[activity],
                record_shapes=True,
            ) as prof:
                out_compiled, code = run_and_get_code(fn_c, *inps)

            prof.export_chrome_trace(trace_path)

            with open(trace_path) as f:
                trace_json = json.load(f)

            triton_events = [
                event
                for event in trace_json["traceEvents"]
                if "triton_poi_fused_0" in event["name"]
            ]
            if torch._inductor.config.combo_kernel_per_subkernel_blocks:
                self.assertEqual([3795, 1, 1], triton_events[0]["args"]["grid"])
            else:
                self.assertEqual([791, 4096, 1], triton_events[0]["args"]["grid"])

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
        if torch._inductor.config.combo_kernel_per_subkernel_blocks:
            FileCheck().check("x_pid_offset = local_pid % x_blocks_0").check(
                "y_pid_offset = local_pid // x_blocks_0"
            ).run(code[0])
        else:
            FileCheck().check("pid_offset = pid").run(code[0])

    @skipIfXpu(msg="Profiler JSON traceEvents is not supported on XPU")
    @requires_gpu_and_triton
    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_combo_kernel_dynamic_shapes_grid_changes(self):
        from torch.profiler import ProfilerActivity

        def fn(x, y):
            return x.sin(), y.cos()

        fn_c = torch.compile(fn)

        def get_grid(x, y):
            with tempfile.NamedTemporaryFile(suffix=".json") as trace_file:
                trace_path = trace_file.name
                activity = getattr(ProfilerActivity, GPU_TYPE.upper())
                with torch.profiler.profile(
                    activities=[activity],
                    record_shapes=True,
                ) as prof:
                    out = fn_c(x, y)
                prof.export_chrome_trace(trace_path)

                with open(trace_path) as f:
                    trace_json = json.load(f)
                triton_events = [
                    e
                    for e in trace_json["traceEvents"]
                    if "triton_poi_fused" in e["name"]
                ]
                return triton_events[0]["args"]["grid"], out

        x1 = torch.randn(1024, 512, device=GPU_TYPE)
        y1 = torch.randn(2048, 256, device=GPU_TYPE)
        grid1, out1 = get_grid(x1, y1)
        eager_out1 = fn(x1, y1)

        x2 = torch.randn(512, 256, device=GPU_TYPE)
        y2 = torch.randn(128, 64, device=GPU_TYPE)
        grid2, out2 = get_grid(x2, y2)
        eager_out2 = fn(x2, y2)

        self.assertNotEqual(grid1[0], grid2[0])
        self.assertEqual(out1, eager_out1)
        self.assertEqual(out2, eager_out2)

        if torch._inductor.config.combo_kernel_per_subkernel_blocks:
            self.assertEqual(grid1[1], 1)
            self.assertEqual(grid2[1], 1)

    @requires_gpu_and_triton
    @parametrize("pointwise_only,expected_kernel_count", [(False, 2), (True, 3)])
    def test_combo_kernels_pointwise_only(self, pointwise_only, expected_kernel_count):
        def fn(a, b, c, d):
            p1 = a * 2.0
            p2 = b + 1.0
            r1 = c.sum(dim=-1)
            r2 = d.mean(dim=-1)
            return p1, p2, r1, r2

        inps = [
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(32, 1024, device=GPU_TYPE),
            torch.rand(32, 1024, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)

        torch._inductor.metrics.reset()
        with torch._inductor.config.patch(
            "combo_kernels_pointwise_only", pointwise_only
        ):
            fn_c = torch.compile(fn)
            out_compiled, _ = run_and_get_code(fn_c, *inps)
            self.assertEqual(out_eager, out_compiled)
            # With pointwise_only=True, we expect more kernels because reductions are not combined with pointwise ops
            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count, expected_kernel_count
            )

    @requires_gpu_and_triton
    @parametrize(
        "max_num_nodes,expected_kernel_count",
        [(8, 1), (3, 2), (2, 3)],
    )
    def test_combo_kernel_max_num_nodes(self, max_num_nodes, expected_kernel_count):
        def fn(a, b, c, d, e, f):
            return (
                a * 2.0,
                b + 1.0,
                c.sin(),
                d.cos(),
                e.exp(),
                f.neg(),
            )

        inps = [
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)

        torch._inductor.metrics.reset()
        with torch._inductor.config.patch("combo_kernel_max_num_nodes", max_num_nodes):
            fn_c = torch.compile(fn)
            out_compiled, _ = run_and_get_code(fn_c, *inps)
            self.assertEqual(out_eager, out_compiled)
            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count, expected_kernel_count
            )

    # waves_per_eu, matrix_instr_nonkdim, and kpack are HIP-only Triton
    # compile options, so only ROCm exercises this combo-kernel rewrite path.
    @unittest.skipIf(not torch.version.hip, "ROCm only")
    @requires_gpu_and_triton
    @parametrize("max_autotune", [False, True])
    def test_combo_kernel_amd_special_config_args(self, max_autotune):
        if not torch._inductor.config.combo_kernel_per_subkernel_blocks:
            self.skipTest("requires combo_kernel_per_subkernel_blocks")

        def fn(a, b):
            return a * 2.0, b + 1.0

        inps = [
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)

        torch._inductor.metrics.reset()
        with torch._inductor.config.patch("max_autotune", max_autotune):
            fn_c = torch.compile(fn)
            out_compiled, _ = run_and_get_code(fn_c, *inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @skipIfXpu(msg="Profiler JSON traceEvents is not supported on XPU")
    @requires_gpu_and_triton
    @unittest.skipIf(not SM90OrLater, "Avoid oom on CI")
    def test_combo_kernel_yz_overflow(self):
        from torch.profiler import ProfilerActivity

        def fn(a, b):
            a_permute = a.permute(0, 2, 1)
            a_clone = a_permute.clone(memory_format=torch.contiguous_format)
            a_view = a_clone.view(-1, a.shape[1])

            b_permute = b.permute(0, 2, 1)
            b_clone = b_permute.clone(memory_format=torch.contiguous_format)
            b_view = b_clone.view(-1, b.shape[1])
            return a_view, b_view

        inps = (
            torch.rand(4800, 34, 256, device=GPU_TYPE),
            torch.rand(22630, 44, 256, device=GPU_TYPE),
        )

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)

        with tempfile.NamedTemporaryFile(suffix=".json") as trace_file:
            trace_path = trace_file.name
            activity = getattr(ProfilerActivity, GPU_TYPE.upper())

            with torch.profiler.profile(
                activities=[activity],
                record_shapes=True,
            ) as prof:
                out_compiled, code = run_and_get_code(fn_c, *inps)

            prof.export_chrome_trace(trace_path)

            with open(trace_path) as f:
                trace_json = json.load(f)

            triton_events = [
                event
                for event in trace_json["traceEvents"]
                if "triton_poi_fused_0" in event["name"]
            ]

            if torch._inductor.config.combo_kernel_per_subkernel_blocks:
                self.assertEqual([83660, 1, 1], triton_events[0]["args"]["grid"])
            else:
                self.assertEqual([4, 45260, 2], triton_events[0]["args"]["grid"])

        self.assertEqual(out_eager, out_compiled)


class ComboKernelBenchmarkTests(TestCase):
    check_model_gpu = check_model_gpu
    check_model_cpu = check_model
    check_kernel_count = True
    combo_kernel_per_subkernel_blocks = False

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                    "combo_kernel_per_subkernel_blocks": self.combo_kernel_per_subkernel_blocks,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    def test_activation_benchmark(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_gpu_and_triton
    def test_reduce_benchmark(self):
        def test_reduce(a, b, c, d):
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10)

    @requires_gpu_and_triton
    def test_mutated_benchmark(self):
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10)

    @requires_gpu_and_triton
    def test_round_robin_dispatch(self):
        # combo kernel dispatch strategy: round robin
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 5, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(5, 18, device=GPU_TYPE),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_gpu_and_triton
    def test_2d_blocking_benchmark(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_gpu(
            fn,
            (
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(40, 30, device=GPU_TYPE),
                torch.rand(36, 40, device=GPU_TYPE),
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(30, 40, device=GPU_TYPE).t(),
                torch.rand(40, 36, device=GPU_TYPE).t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_gpu_and_triton
    def test_persistent_reduction_no_x_dim(self):
        def fn(x, y):
            return x.sum(1), y.sum(1)

        inps = (
            torch.rand(16, 256, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)


class ComboKernelDynamicShapesTests(TestCase):
    check_model_gpu = check_model_gpu
    check_model_cpu = check_model
    check_kernel_count = True
    combo_kernel_per_subkernel_blocks = False

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                    "combo_kernel_per_subkernel_blocks": self.combo_kernel_per_subkernel_blocks,
                }
            )
        )
        self._test_stack.enter_context(
            torch._dynamo.config.patch(
                {
                    "automatic_dynamic_shapes": False,
                    "assume_static_by_default": False,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    def test_dynamic_shapes_activations(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_gpu_and_triton
    def test_dynamic_shapes_2d_blocking(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_gpu(
            fn,
            (
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(40, 30, device=GPU_TYPE),
                torch.rand(36, 40, device=GPU_TYPE),
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(30, 40, device=GPU_TYPE).t(),
                torch.rand(40, 36, device=GPU_TYPE).t(),
            ),
        )
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_gpu_and_triton
    def test_dynamic_shapes_reduce(self):
        def test_reduce(a, b, c, d):
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10)

    @requires_gpu_and_triton
    def test_dynamic_shapes_mutated(self):
        # combo kernel dispatch strategy: round robin
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 5, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(5, 18, device=GPU_TYPE),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_gpu_and_triton
    @torch._inductor.config.patch("combo_kernels_autotune", 0)
    def test_dynamic_shapes_activations_no_autotune(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_gpu_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_persistent_reduction_no_x_dim(self):
        def fn(x, y):
            return x.sum(1), y.sum(1)

        inps = (
            torch.rand(16, 256, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)

    @requires_gpu_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_persistent_reduction_no_x_dim_2(self):
        def fn(x, y):
            return x.sum(2), y.sum(2)

        inps = (
            torch.rand(8, 16, 256, device=GPU_TYPE),
            torch.rand(8, 32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], (0, 1), min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], (0, 1), min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)

    @requires_gpu_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_2d_blocking_round_robin(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        inps = (
            torch.rand(20, 30, device=GPU_TYPE),
            torch.rand(30, 30, device=GPU_TYPE),
            torch.rand(40, 32, device=GPU_TYPE),
            torch.rand(30, 20, device=GPU_TYPE).t(),
            torch.rand(30, 30, device=GPU_TYPE).t(),
            torch.rand(32, 40, device=GPU_TYPE).t(),
        )

        out_eager = fn(*inps)
        compiled = torch.compile(fn)
        out_compiled = compiled(*inps)
        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(5 <= torch._inductor.metrics.generated_kernel_count <= 6)
        torch._inductor.metrics.reset()

        inps = (
            torch.rand(24, 30, device=GPU_TYPE),
            torch.rand(32, 30, device=GPU_TYPE),
            torch.rand(48, 32, device=GPU_TYPE),
            torch.rand(30, 24, device=GPU_TYPE).t(),
            torch.rand(30, 32, device=GPU_TYPE).t(),
            torch.rand(32, 48, device=GPU_TYPE).t(),
        )
        out_compiled = compiled(*inps)
        out_eager = fn(*inps)
        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(5 <= torch._inductor.metrics.generated_kernel_count <= 6)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    @torch._inductor.config.patch("triton.autotune_at_compile_time", True)
    def test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda(self):
        def fn(x, y, z):
            return x.sum(1), y.mean(1), z.max(1)

        inps = (
            torch.rand(16, 128, device=GPU_TYPE),
            torch.rand(32, 128, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[2], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)

    @requires_xpu_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    @torch._inductor.config.patch("triton.autotune_at_compile_time", True)
    def test_dynamic_shapes_persistent_reduction_mixed_x_dim_xpu(self):
        def fn(x, y, z):
            return x.sum(1), y.mean(1), z.max(1)

        inps = (
            torch.rand(16, 128, device=GPU_TYPE),
            torch.rand(32, 128, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[2], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)

    @requires_gpu_and_triton
    def test_helper_fn_defined(self):
        def fn(x, y, z):
            return x.sum(1), y.mean(1), z.cumsum(1)

        inps = (
            torch.rand(16, 128, device=GPU_TYPE),
            torch.rand(32, 128, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        code = " ".join(code)
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(code.count("def _triton_helper_fn_add0(arg0_0, arg1_0):"), 1)


class ComboKernelTestsPerSubkernelBlocks(ComboKernelTests):
    combo_kernel_per_subkernel_blocks = True


class ComboKernelBenchmarkTestsPerSubkernelBlocks(ComboKernelBenchmarkTests):
    combo_kernel_per_subkernel_blocks = True


class ComboKernelDynamicShapesTestsPerSubkernelBlocks(ComboKernelDynamicShapesTests):
    combo_kernel_per_subkernel_blocks = True


@instantiate_parametrized_tests
class ComboKernelPDLTests(TestCase):
    """Tests for PDL (Programmatic Dependent Launch) support in combo kernels."""

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                    "triton.enable_pdl": True,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    @skipIfRocm
    @unittest.skipIf(not SM90OrLater, "PDL requires SM90 or later (Hopper+)")
    def test_pdl_codegen_in_combo_kernel(self):
        """Test that PDL flag and gdc calls are generated in combo kernels."""

        def fn(a, b):
            return torch.relu(a), torch.sigmoid(b)

        inps = [
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
        ]

        fn_c = torch.compile(fn)
        _, code = run_and_get_code(fn_c, *inps)
        code = " ".join(code)

        # Check that launch_pdl is True and PDL API calls are generated
        FileCheck().check("'launch_pdl': True").run(code)

        # Each sub-kernel should have exactly one gdc_wait followed by one
        # gdc_launch_dependents, with no redundant waits in between.
        # Uses round-robin dispatch (pid % 2) since both tensors are same size.
        (
            FileCheck()
            .check("if pid")
            .check("tl.extra.cuda.gdc_wait()")
            .check("tl.load(")
            .check_not("tl.extra.cuda.gdc_wait()")
            .check("tl.extra.cuda.gdc_launch_dependents()")
            .check_not("tl.extra.cuda.gdc_wait()")
            .check("elif pid % 2 == 1:")
            .check("tl.extra.cuda.gdc_wait()")
            .check("tl.load(")
            .check_not("tl.extra.cuda.gdc_wait()")
            .check("tl.extra.cuda.gdc_launch_dependents()")
            .run(code)
        )

    @requires_gpu_and_triton
    @skipIfRocm
    @unittest.skipIf(not SM90OrLater, "PDL requires SM90 or later (Hopper+)")
    def test_pdl_combo_kernel_pointwise(self):
        """Test that pointwise combo kernels produce correct results with PDL."""

        def fn(a, b, c):
            return torch.relu(a), torch.sigmoid(b), torch.tanh(c)

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        code = " ".join(code)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

        # Verify combo kernel structure with PDL - each sub-kernel should have
        # exactly one gdc_wait and one gdc_launch_dependents, no redundant waits.
        (
            FileCheck()
            .check("'launch_pdl': True")
            .check("if pid < num_xblocks_0:")
            .check("tl.extra.cuda.gdc_wait()")
            .check_not("tl.extra.cuda.gdc_wait()")
            .check("tl.extra.cuda.gdc_launch_dependents()")
            .check("elif pid < num_xblocks_1:")
            .check("tl.extra.cuda.gdc_wait()")
            .check_not("tl.extra.cuda.gdc_wait()")
            .check("tl.extra.cuda.gdc_launch_dependents()")
            .check("elif pid < num_xblocks_2:")
            .check("tl.extra.cuda.gdc_wait()")
            .check_not("tl.extra.cuda.gdc_wait()")
            .check("tl.extra.cuda.gdc_launch_dependents()")
            .run(code)
        )

    @requires_gpu_and_triton
    @skipIfRocm
    @unittest.skipIf(not SM90OrLater, "PDL requires SM90 or later (Hopper+)")
    def test_pdl_combo_kernel_reduction(self):
        """Test that reduction combo kernels produce correct results with PDL."""

        def fn(x, y):
            return x.sum(dim=-1), y.mean(dim=-1)

        inps = [
            torch.rand(32, 1024, device=GPU_TYPE),
            torch.rand(32, 1024, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)


class ComboKernelTestsMaxAutotune(TestCase):
    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                    "combo_kernel_per_subkernel_blocks": True,
                    "max_autotune": True,
                    "autotune_local_cache": False,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    def test_combo_kernel_max_autotune(self):
        def fn(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(32, 1024, device=GPU_TYPE),
            torch.rand(64, 512, device=GPU_TYPE),
            torch.rand(16, 2048, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)

        logger = logging.getLogger("torch._inductor.runtime.triton_heuristics")
        with self.assertLogs(logger, level=logging.DEBUG) as cm:
            out_compiled, code = run_and_get_code(fn_c, *inps)
        chained_logs = [msg for msg in cm.output if "Combo sequential autotune" in msg]
        self.assertGreater(
            len(chained_logs),
            0,
            "_combo_sequential_autotune was not invoked",
        )
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    def test_combo_kernel_max_autotune_with_reduction(self):
        def fn(x, y):
            return x.sum(dim=-1), y.mean(dim=-1)

        inps = [
            torch.rand(128, 256, device=GPU_TYPE),
            torch.rand(128, 256, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)

        logger = logging.getLogger("torch._inductor.runtime.triton_heuristics")
        with self.assertLogs(logger, level=logging.DEBUG) as cm:
            out_compiled, code = run_and_get_code(fn_c, *inps)
        chained_logs = [msg for msg in cm.output if "Combo sequential autotune" in msg]
        self.assertGreater(
            len(chained_logs),
            0,
            "_combo_sequential_autotune was not invoked",
        )
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    def test_combo_autotune_many_subkernels(self):
        def fn(a, b, c, d, e, f):
            return (
                a * 2.0,
                b + 1.0,
                c.sin(),
                d.cos(),
                e.exp(),
                f.neg(),
            )

        inps = [
            torch.rand(8, 8192, device=GPU_TYPE),
            torch.rand(128, 64, device=GPU_TYPE),
            torch.rand(16, 4096, device=GPU_TYPE),
            torch.rand(512, 16, device=GPU_TYPE),
            torch.rand(32, 2048, device=GPU_TYPE),
            torch.rand(256, 32, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)

        logger = logging.getLogger("torch._inductor.runtime.triton_heuristics")
        with self.assertLogs(logger, level=logging.DEBUG) as cm:
            out_compiled, code = run_and_get_code(fn_c, *inps)

        chained_logs = [msg for msg in cm.output if "Combo sequential autotune" in msg]
        self.assertGreater(len(chained_logs), 0)
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    def test_combo_kernel_per_subkernel_reduction_hint(self):
        def fn(x, y):
            return x.sum(dim=-1), y.sum(dim=0)

        inps = [
            torch.rand(128, 256, device=GPU_TYPE),
            torch.rand(128, 256, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        out, code = run_and_get_code(torch.compile(fn), *inps)
        self.assertEqual(out_eager, out)
        # Verify per-subkernel reduction hints in generated code
        found_hints = {}
        for c in code:
            for key in ["reduction_hint_0", "reduction_hint_1"]:
                m = re.search(rf"'{key}':\s*'(\w+)'", c)
                if m:
                    found_hints[key] = m.group(1)

        self.assertIn(
            "reduction_hint_0", found_hints, "Missing per-subkernel reduction_hint_0"
        )
        self.assertIn(
            "reduction_hint_1", found_hints, "Missing per-subkernel reduction_hint_1"
        )
        self.assertEqual(found_hints["reduction_hint_0"], "INNER")
        self.assertEqual(found_hints["reduction_hint_1"], "OUTER")

    @requires_gpu_and_triton
    @torch._inductor.config.patch("combo_kernel_autotune_grouping", True)
    def test_combo_autotune_grouping(self):
        def fn(a, b, c, d):
            return a.cos(), b.sin(), c.exp(), d.neg()

        # a,b: numel=262144 → bs=1024, c,d: numel=32 → bs=256
        # Different bs → different configs → separate groups
        inps = [
            torch.rand(4, 65536, device=GPU_TYPE),
            torch.rand(4, 65536, device=GPU_TYPE),
            torch.rand(4, 8, device=GPU_TYPE),
            torch.rand(4, 8, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)

        logger = logging.getLogger("torch._inductor.runtime.triton_heuristics")
        with self.assertLogs(logger, level=logging.DEBUG) as cm:
            out_compiled, code = run_and_get_code(fn_c, *inps)

        # Parse "Phase 1 group N SK[...]" lines to check grouping
        group_lines = [
            msg for msg in cm.output if "Phase 1 group" in msg and "SK[" in msg
        ]
        group_indices = {
            int(re.search(r"group (\d+)", line).group(1))
            for line in group_lines
            if re.search(r"group (\d+)", line)
        }
        # Exact grouping count is hardware-dependent because pointwise candidate
        # config sets can differ across environments. The stable regression for
        # the new grouping key lives in the mocked test below.
        self.assertGreater(
            len(group_indices),
            0,
            f"Expected at least one autotune group, got {group_lines}",
        )
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    @torch._inductor.config.patch("combo_kernel_autotune_grouping", True)
    def test_combo_autotune_grouping_uses_tiling_signature(self):
        import triton

        inductor_meta = {
            "combo_grid_meta": {
                "num_kernels": 2,
                "heuristic_0": "pointwise",
                "heuristic_1": "pointwise",
                "size_hints_0": {"x": 256, "y": 256},
                "size_hints_1": {"x": 256, "y": 256},
                "tile_hint_0": "TileHint.SQUARE",
                "tile_hint_1": "TileHint.SQUARE",
                "tiling_scores_0": {"x": 8, "y": 1},
                "tiling_scores_1": {"x": 1, "y": 8},
            }
        }

        def pointwise_configs(*args, **kwargs):
            return [
                triton.Config({"XBLOCK": 64, "YBLOCK": 32}, num_warps=4, num_stages=1),
                triton.Config({"XBLOCK": 128, "YBLOCK": 32}, num_warps=4, num_stages=1),
            ]

        with unittest.mock.patch(
            "torch._inductor.runtime.triton_heuristics.pointwise",
            side_effect=pointwise_configs,
        ):
            torch._inductor.runtime.triton_heuristics._handle_combo_kernel_per_subkernel_blocks(
                {"x": 256, "y": 256},
                inductor_meta,
                triton_meta={},
            )

        groups = inductor_meta["combo_tuning_groups"]
        self.assertEqual(len(groups), 2)
        self.assertEqual([[0], [1]], [g["member_indices"] for g in groups])

    @requires_gpu_and_triton
    def test_combo_kernel_coordesc_tunes_largest_subkernel_first(self):
        def fn(a, b, c):
            return (
                torch.nn.functional.relu(a),
                torch.nn.functional.sigmoid(b),
                torch.nn.functional.tanh(c),
            )

        inps = [
            torch.rand(32, 1024, device=GPU_TYPE),
            torch.rand(256, 256, device=GPU_TYPE),
            torch.rand(16, 128, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)

        def parse_block_cfg(msg: str) -> dict[str, int]:
            return {
                m.group(1): int(m.group(2))
                for m in re.finditer(r"(\w+BLOCK_\d+): (\d+)", msg)
            }

        logger = logging.getLogger("torch._inductor.runtime.coordinate_descent_tuner")
        with torch._inductor.config.patch(coordinate_descent_tuning=True):
            with self.assertLogs(logger, level=logging.DEBUG) as cm:
                out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)

        baseline_log = next(
            msg for msg in cm.output if "Baseline Config" in msg and "XBLOCK_" in msg
        )
        baseline_cfg = parse_block_cfg(baseline_log)
        try_logs = [
            msg for msg in cm.output if "Try config" in msg and "XBLOCK_" in msg
        ]
        self.assertGreater(
            len(try_logs), 0, "Coordinate descent did not try combo fields"
        )
        distinct_block_cfgs = {
            tuple(sorted(parse_block_cfg(msg).items())) for msg in try_logs
        }
        self.assertGreater(
            len(distinct_block_cfgs),
            1,
            "Coordinate descent did not explore different suffixed block sizes.",
        )

        first_cfg = parse_block_cfg(try_logs[0])
        changed_fields = {
            key for key, value in first_cfg.items() if baseline_cfg.get(key) != value
        }
        self.assertEqual(
            changed_fields,
            {"XBLOCK_1"},
            f"Expected the first combo coordesc step to tune the largest subkernel first, got {changed_fields}",
        )


@instantiate_parametrized_tests
class ComboKernelMetadataTests(TestCase):
    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                    "combo_kernel_per_subkernel_blocks": True,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    def _combo_code(self, fn, inps):
        out_eager = fn(*inps)
        out_compiled, code = run_and_get_code(torch.compile(fn), *inps)
        self.assertEqual(out_eager, out_compiled)
        return " ".join(code)

    @requires_gpu_and_triton
    def test_combo_inductor_meta_has_optimize_mem(self):
        def fn(a, b):
            return torch.relu(a), torch.sigmoid(b)

        inps = [torch.rand(1024, device=GPU_TYPE) for _ in range(2)]
        code = self._combo_code(fn, inps)
        self.assertIn("'optimize_mem': True", code)

    @requires_gpu_and_triton
    def test_combo_inductor_meta_optimize_mem_false_in_training_forward(self):
        def fn(a, b):
            return torch.relu(a), torch.sigmoid(b)

        inps = [torch.rand(1024, device=GPU_TYPE, requires_grad=True) for _ in range(2)]
        code = self._combo_code(fn, inps)
        self.assertIn("'optimize_mem': False", code)

    @requires_gpu_and_triton
    @parametrize("disable_ftz", [False, True])
    def test_combo_triton_meta_has_disable_ftz(self, disable_ftz):
        def fn(a, b):
            return torch.relu(a), torch.sigmoid(b)

        inps = [torch.rand(1024, device=GPU_TYPE) for _ in range(2)]
        with torch._inductor.config.patch({"eager_numerics.disable_ftz": disable_ftz}):
            code = self._combo_code(fn, inps)
        self.assertIn(f"'disable_ftz': {disable_ftz}", code)


# Minimal scheduler doubles for direct _try_combo_with_memory_check tests.
class _PeakMemFakeNode:
    def __init__(self, name: str, deps=()) -> None:
        self.name = name
        self.scheduler = object()
        self.unmet_dependencies = tuple(SimpleNamespace(name=dep) for dep in deps)

    def get_name(self) -> str:
        return self.name

    def get_buffer_names(self):
        return ()


class _PeakMemFakeBuffer:
    def __init__(self, name: str, defining_op: _PeakMemFakeNode, succ_nodes) -> None:
        self.name = name
        self.defining_op = defining_op
        self.mpi_buffer = SimpleNamespace(succ_nodes=succ_nodes)

    def get_name(self) -> str:
        return self.name


class _PeakMemFakeFreeableInputBuffer:
    pass


class _PeakMemFakeScheduler:
    def __init__(self, nodes, name_to_fused_node=None) -> None:
        self.nodes = nodes
        self.name_to_fused_node = (
            {} if name_to_fused_node is None else name_to_fused_node
        )

    def topological_sort_schedule(self, nodes):
        return nodes


class ComboKernelPeakMemoryTests(InductorTestCase):
    """Coverage for memory-aware combo-kernel acceptance and commit logic."""

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                    "combo_kernel_per_subkernel_blocks": True,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @staticmethod
    def _thresholds(*, abs_thr=None, pct_thr=None):
        return {
            "combo_kernel_peak_memory_threshold": abs_thr,
            "combo_kernel_peak_memory_pct_threshold": pct_thr,
        }

    @staticmethod
    def _make_wide_resnet_like():
        """Build a WideResNet-like model."""

        class Bottleneck(torch.nn.Module):
            expansion = 4

            def __init__(self, in_ch, mid_ch, stride=1, downsample=None):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_ch, mid_ch, 1, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(mid_ch)
                self.conv2 = torch.nn.Conv2d(
                    mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False
                )
                self.bn2 = torch.nn.BatchNorm2d(mid_ch)
                self.conv3 = torch.nn.Conv2d(
                    mid_ch, mid_ch * self.expansion, 1, bias=False
                )
                self.bn3 = torch.nn.BatchNorm2d(mid_ch * self.expansion)
                self.relu = torch.nn.ReLU(inplace=True)
                self.downsample = downsample

            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                if self.downsample is not None:
                    identity = self.downsample(x)
                return self.relu(out + identity)

        def make_layer(in_ch, mid_ch, blocks, stride=1):
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, mid_ch * 4, 1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(mid_ch * 4),
            )
            layers = [Bottleneck(in_ch, mid_ch, stride, downsample)]
            for _ in range(1, blocks):
                layers.append(Bottleneck(mid_ch * 4, mid_ch))
            return torch.nn.Sequential(*layers)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(64)
                self.relu = torch.nn.ReLU(inplace=True)
                self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
                self.layer1 = make_layer(64, 128, blocks=3)
                self.layer2 = make_layer(512, 256, blocks=4, stride=2)
                self.layer3 = make_layer(1024, 512, blocks=20, stride=2)
                self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
                self.fc = torch.nn.Linear(2048, 1000)

            def forward(self, x):
                x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                return self.fc(self.avgpool(x).flatten(1))

        return Model()

    @staticmethod
    def _make_wide_resnet_case():
        # This model shape reliably exercises the combo-kernel search paths used
        # by the end-to-end memory-aware tests.
        model = ComboKernelPeakMemoryTests._make_wide_resnet_like().to(GPU_TYPE).eval()
        x = torch.randn(1, 3, 224, 224, device=GPU_TYPE)
        with torch.no_grad():
            out_eager = model(x)
        return model, x, out_eager

    @staticmethod
    def _make_peak_mem_context(
        nodes,
        *,
        buf_info_list,
        memories_at_nodes,
        accepted_step=None,
    ):
        from torch._inductor.scheduler import ComboKernelMemoryContext

        return ComboKernelMemoryContext(
            graph_outputs=set(),
            buf_info_list=buf_info_list,
            freeable_input_buffer_cls=_PeakMemFakeFreeableInputBuffer,
            memories_at_nodes=memories_at_nodes,
            node_to_idx={node: idx for idx, node in enumerate(nodes)},
            accepted_step={} if accepted_step is None else dict(accepted_step),
        )

    @staticmethod
    def _try_combo_with_fake_scheduler(
        nodes,
        group_nodes,
        *,
        buf_info_list,
        baseline_peak,
        memories_at_nodes,
        combo_name="combo",
        thresholds,
        name_to_fused_node=None,
        accepted_step=None,
    ):
        from unittest.mock import patch

        from torch._inductor.scheduler import Scheduler

        scheduler = _PeakMemFakeScheduler(nodes, name_to_fused_node)
        # Unit tests pass a precomputed baseline timeline to isolate the local
        # combo logic from the full scheduler setup path.
        mem_ctx = ComboKernelPeakMemoryTests._make_peak_mem_context(
            nodes,
            buf_info_list=buf_info_list,
            memories_at_nodes=memories_at_nodes,
            accepted_step=accepted_step,
        )

        def fake_foreach(*args, **kwargs):
            return _PeakMemFakeNode(combo_name)

        with patch(
            "torch._inductor.scheduler.ForeachKernelSchedulerNode", fake_foreach
        ):
            with torch._inductor.config.patch(**thresholds):
                return Scheduler._try_combo_with_memory_check(
                    scheduler,
                    group_nodes,
                    mem_ctx,
                    baseline_peak=baseline_peak,
                    enable_autotune=False,
                )

    def test_threshold_gating(self):
        """abs_thr/pct_thr set to 0 or a too-small bound reject; default accepts."""
        a = _PeakMemFakeNode("a")
        consume_a = _PeakMemFakeNode("consume_a", deps=("buf_a",))
        b = _PeakMemFakeNode("b")
        consume_b = _PeakMemFakeNode("consume_b", deps=("buf_b",))
        nodes = [a, consume_a, b, consume_b]
        buf_info_list = [
            BufferInfo(_PeakMemFakeBuffer("buf_a", a, {consume_a}), 100, 100, 0, 1),
            BufferInfo(_PeakMemFakeBuffer("buf_b", b, {consume_b}), 100, 100, 2, 3),
        ]
        memories_at_nodes = [100, 100, 100, 100, 0]

        def run(thresholds):
            return self._try_combo_with_fake_scheduler(
                nodes,
                [a, b],
                buf_info_list=buf_info_list,
                baseline_peak=100,
                memories_at_nodes=memories_at_nodes,
                thresholds=thresholds,
            )

        # Rejection cases: any limit below the +100 delta the combo forces.
        for label, thresholds in (
            ("abs=0", self._thresholds(abs_thr=0)),
            ("pct=0", self._thresholds(pct_thr=0.0)),
            ("abs=1", self._thresholds(abs_thr=1)),
        ):
            combo, *_ = run(thresholds)
            self.assertIsNone(combo, f"{label} should reject")

        # Both thresholds disabled → accept and surface the new region timeline.
        combo, _, new_region_memory, region_start, combo_step = run(self._thresholds())
        self.assertIsNotNone(combo)
        self.assertEqual(region_start, 0)
        self.assertEqual(combo_step, 0)
        self.assertEqual(new_region_memory, [200, 200, 100, 100])

    def test_commit_semantics(self):
        """Three invariants of _commit_combo_to_memory_context:
        (a) it only rewrites [region_start, region_end+1], leaving the suffix intact;
        (b) it correctly handles overlapping commits"""
        from torch._inductor.scheduler import ComboKernelMemoryContext, Scheduler

        # (a) Single commit preserves the suffix outside the region.
        a = _PeakMemFakeNode("a")
        b = _PeakMemFakeNode("b")
        c = _PeakMemFakeNode("c")
        later_user = _PeakMemFakeNode("later_user")
        nodes = [a, b, c, later_user]
        buf = _PeakMemFakeBuffer("buf", b, {later_user})
        mem_ctx = self._make_peak_mem_context(
            nodes,
            buf_info_list=[BufferInfo(buf, 10, 10, 1, 3)],
            memories_at_nodes=[0, 10, 10, 10, 0],
        )
        combo_node, _, new_region_memory, region_start, combo_step = (
            self._try_combo_with_fake_scheduler(
                nodes,
                [a, b],
                buf_info_list=mem_ctx.buf_info_list,
                baseline_peak=10,
                memories_at_nodes=mem_ctx.memories_at_nodes,
                thresholds=self._thresholds(),
            )
        )
        self.assertEqual(new_region_memory, [10, 10, 10])
        Scheduler._commit_combo_to_memory_context(
            None,  # type: ignore[arg-type]
            mem_ctx,
            region_start,
            combo_node,
            combo_step,
            new_region_memory,
            [a, b],
        )
        # Suffix (index 4) stays at its pre-commit value of 0.
        self.assertEqual(mem_ctx.memories_at_nodes, [10, 10, 10, 10, 0])

        # (b) Overlapping commits: combo1 writes [4..11], combo2 writes [8..15].
        # This can't be produced reliably end-to-end so we feed it directly.
        N = 20
        mem_ctx = ComboKernelMemoryContext(
            graph_outputs=set(),
            buf_info_list=[],
            freeable_input_buffer_cls=_PeakMemFakeFreeableInputBuffer,
            memories_at_nodes=[100] * (N + 1),
            node_to_idx={},
        )
        combo1_nodes = [_PeakMemFakeNode(f"a{i}") for i in range(5)]
        Scheduler._commit_combo_to_memory_context(
            None,  # type: ignore[arg-type]
            mem_ctx,
            region_start=4,
            combo_node=_PeakMemFakeNode("combo1"),
            combo_step=5,
            new_region_memory=[120, 150, 200, 200, 180, 160, 140, 130],
            group_nodes=combo1_nodes,
        )
        combo2_nodes = [_PeakMemFakeNode(f"b{i}") for i in range(4)]
        Scheduler._commit_combo_to_memory_context(
            None,  # type: ignore[arg-type]
            mem_ctx,
            region_start=8,
            combo_node=_PeakMemFakeNode("combo2"),
            combo_step=9,
            new_region_memory=[170, 190, 210, 195, 175, 155, 145, 135],
            group_nodes=combo2_nodes,
        )
        self.assertEqual(max(mem_ctx.memories_at_nodes), 210)
        # Spot-check: combo1-only prefix, overlap (combo2 wins), combo2-only,
        # untouched suffix.
        self.assertEqual(mem_ctx.memories_at_nodes[4], 120)
        self.assertEqual(mem_ctx.memories_at_nodes[7], 200)
        self.assertEqual(mem_ctx.memories_at_nodes[8], 170)
        self.assertEqual(mem_ctx.memories_at_nodes[11], 195)
        self.assertEqual(mem_ctx.memories_at_nodes[15], 135)
        self.assertEqual(mem_ctx.memories_at_nodes[16], 100)
        self.assertEqual(mem_ctx.last_region_start, 8)
        for n in combo1_nodes:
            self.assertEqual(mem_ctx.accepted_step[n], 5)
        for n in combo2_nodes:
            self.assertEqual(mem_ctx.accepted_step[n], 9)

    @requires_gpu_and_triton
    def test_combo_kernel_peak_memory_wide_resnet(self):
        """A tight peak-memory threshold must measurably reduce
        the runtime CUDA peak memory of the compiled forward pass compared to
        the gating-disabled baseline"""
        model, x, out_eager = self._make_wide_resnet_case()

        def compile_and_measure_peak(**cfg):
            torch._dynamo.reset()
            torch._inductor.metrics.reset()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with fresh_cache(), torch._inductor.config.patch(**cfg):
                with torch.no_grad():
                    out = torch.compile(model)(x)
                torch.cuda.synchronize()
            self.assertEqual(out_eager, out)
            return torch.cuda.max_memory_allocated()

        # Gating disabled: combo fusion is free to co-allocate outputs, which
        # inflates peak memory. combo_kernel_max_distance has no effect on
        # this path since _distance_windows is only called when gating is on.
        peak_disabled = compile_and_measure_peak(
            **self._thresholds(abs_thr=None, pct_thr=None),
        )

        # Tight abs threshold: reject any combo that would increase peak.
        peak_tight = compile_and_measure_peak(
            **self._thresholds(abs_thr=1 << 20),
        )

        # tight threshold produces ~50 MB peak reduction on WideResNet forward
        self.assertLess(
            peak_tight,
            peak_disabled,
            f"tight threshold did not reduce runtime peak memory "
            f"(tight={peak_tight}, disabled={peak_disabled})",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU or HAS_GPU_AND_TRITON:
        run_tests(needs="filelock")
