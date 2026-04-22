# Owner(s): ["module: inductor"]
import os
import unittest
from collections.abc import Callable
from unittest import mock

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.template_heuristics.triton import (
    GemmConfig,
    ROCmAddMMTemplateConfigHeuristic,
    ROCmMMTemplateConfigHeuristic,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"
ORIGAMI_TOPK = 5
ORIGAMI_COMPILE_TOPK = 2  # must be < candidate pool size (4) in _make_heuristic
PERF_SLOWDOWN_TOLERANCE = 1.15
IS_ROCM = torch.version.hip is not None

try:
    import origami

    HAS_ORIGAMI = True
except ImportError:
    origami = None
    HAS_ORIGAMI = False


if IS_ROCM:
    torch.set_float32_matmul_precision("highest")


@unittest.skipIf(not HAS_GPU_AND_TRITON, "requires GPU and Triton")
@unittest.skipIf(not IS_ROCM, "Origami integration is ROCm-only")
@unittest.skipIf(not HAS_ORIGAMI, "Origami package is not installed")
class TestOrigami(TestCase):
    def _make_heuristic(self, op_name: str):
        if op_name == "addmm":
            heuristic = ROCmAddMMTemplateConfigHeuristic()
        else:
            heuristic = ROCmMMTemplateConfigHeuristic()

        # Keep the test search space intentionally small and valid so we can
        # compare Origami's filtered path against exhaustive autotuning on the
        # exact same candidate pool.
        configs = [
            GemmConfig(64, 64, 32, 2, 4, group_m=8),
            GemmConfig(64, 128, 32, 3, 4, group_m=8),
            GemmConfig(128, 64, 32, 3, 4, group_m=8),
            GemmConfig(128, 128, 64, 3, 4, group_m=8),
        ]
        heuristic.mm_configs = configs
        heuristic.exhaustive_configs = configs
        return heuristic

    def _make_fn_and_inputs(
        self, op_name: str, size: int
    ) -> tuple[Callable[..., torch.Tensor], tuple[torch.Tensor, ...]]:
        torch.manual_seed(0)
        a = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float16)

        if op_name == "mm":

            def fn(x, y):
                return torch.mm(x, y)

            return fn, (a, b)

        if op_name == "addmm":
            bias = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float16)

            def fn(inp, x, y):
                return torch.addmm(inp, x, y)

            return fn, (bias, a, b)

        raise AssertionError(f"Unsupported op {op_name}")

    def _benchmark_gpu_call_count(self) -> int:
        return sum(
            value
            for name, value in counters["inductor"].items()
            if "benchmark_gpu" in name
        )

    def _compile_with_config(
        self,
        op_name: str,
        patch_config: dict[str, object],
        *,
        size: int,
    ) -> dict[str, object]:
        fn, args = self._make_fn_and_inputs(op_name, size)
        expected = fn(*args)
        heuristic = self._make_heuristic(op_name)

        torch._dynamo.reset()
        counters.clear()

        with (
            fresh_cache(),
            config.patch(patch_config),
            mock.patch(
                "torch._inductor.template_heuristics.registry.get_template_heuristic",
                return_value=heuristic,
            ),
            mock.patch(
                "origami.select_topk_configs", wraps=origami.select_topk_configs
            ) as select_topk,
        ):
            compiled = torch.compile(fn, dynamic=False)
            result = compiled(*args)

        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

        return {
            "compiled": compiled,
            "args": args,
            "benchmark_gpu_calls": self._benchmark_gpu_call_count(),
            "topk_calls": select_topk.call_count,
        }

    def _origami_default_config(self) -> dict[str, object]:
        return {
            "max_autotune": False,
            "max_autotune_gemm": True,
            "origami": True,
            "origami_topk": ORIGAMI_TOPK,
            "max_autotune_gemm_search_space": "DEFAULT",
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "test_configs.autotune_choice_name_regex": r"^triton_mm_",
            "triton.native_matmul": False,
        }

    def _origami_exhaustive_config(self) -> dict[str, object]:
        return {
            "max_autotune": False,
            "max_autotune_gemm": True,
            "origami": True,
            "origami_topk": ORIGAMI_TOPK,
            "max_autotune_gemm_search_space": "EXHAUSTIVE",
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "test_configs.autotune_choice_name_regex": r"^triton_mm_",
            "triton.native_matmul": False,
        }

    def _max_autotune_default_config(self) -> dict[str, object]:
        return {
            "max_autotune": False,
            "max_autotune_gemm": True,
            "origami": False,
            "max_autotune_gemm_search_space": "DEFAULT",
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "test_configs.autotune_choice_name_regex": r"^triton_mm_",
            "triton.native_matmul": False,
        }

    def test_origami_respects_gemm_search_space(self):
        for op_name in ("mm", "addmm"):
            with self.subTest(op_name=op_name, search_space="DEFAULT"):
                default_case = self._compile_with_config(
                    op_name,
                    self._origami_default_config(),
                    size=256,
                )
                self.assertGreater(default_case["topk_calls"], 0)

            with self.subTest(op_name=op_name, search_space="EXHAUSTIVE"):
                exhaustive_case = self._compile_with_config(
                    op_name,
                    self._origami_exhaustive_config(),
                    size=256,
                )
                self.assertEqual(exhaustive_case["topk_calls"], 0)

    def test_origami_reduces_compile_work_vs_regular_max_autotune(self):
        for op_name in ("mm", "addmm"):
            with self.subTest(op_name=op_name):
                origami_case = self._compile_with_config(
                    op_name,
                    {
                        **self._origami_default_config(),
                        "origami_topk": ORIGAMI_COMPILE_TOPK,
                    },
                    size=256,
                )
                max_autotune_case = self._compile_with_config(
                    op_name,
                    self._max_autotune_default_config(),
                    size=256,
                )

                # Benchmark call count is a deterministic proxy for autotune
                # compile work: Origami should benchmark fewer candidate GEMMs
                # than regular max-autotune on the same candidate pool.
                self.assertEqual(max_autotune_case["topk_calls"], 0)
                self.assertLess(
                    origami_case["benchmark_gpu_calls"],
                    max_autotune_case["benchmark_gpu_calls"],
                )

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_origami_runtime_matches_regular_max_autotune(self):
        for op_name in ("mm", "addmm"):
            for size in (1024, 8192, 16384):
                with self.subTest(op_name=op_name, size=size):
                    origami_case = self._compile_with_config(
                        op_name,
                        self._origami_default_config(),
                        size=size,
                    )
                    max_autotune_case = self._compile_with_config(
                        op_name,
                        self._max_autotune_default_config(),
                        size=size,
                    )

                    origami_runtime_ms = benchmarker.benchmark(
                        origami_case["compiled"],
                        origami_case["args"],
                        {},
                        warmup=50,
                        rep=200,
                    )
                    max_autotune_runtime_ms = benchmarker.benchmark(
                        max_autotune_case["compiled"],
                        max_autotune_case["args"],
                        {},
                        warmup=50,
                        rep=200,
                    )

                    print(
                        f"{op_name} size={size} runtime ms: origami={origami_runtime_ms:.3f}, "
                        f"max_autotune={max_autotune_runtime_ms:.3f}"
                    )

                    self.assertLessEqual(
                        origami_runtime_ms,
                        max_autotune_runtime_ms * PERF_SLOWDOWN_TOLERANCE,
                    )


if __name__ == "__main__":
    if HAS_GPU_AND_TRITON and IS_ROCM:
        run_tests()
