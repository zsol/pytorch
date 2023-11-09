# Owner(s): ["module: inductor"]

from typing import List

import torch
from torch._C import FileCheck
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import load_test_module
from torch._dynamo.utils import same
from torch._inductor import config

from torch.testing._internal.common_utils import skipIfRocm
from torch.testing._internal.inductor_utils import run_and_get_cpp_code


@config.patch(memory_planning=True)
class TestMemoryPlanning(TestCase):
    def _generate(self, *, device):
        """
        Generate a simple test case that has multiple simultaneously-live intermediate tensors.
        """

        def f(x, y, z):
            t0 = x.matmul(y)
            t1 = x.matmul(z)
            t0 = x.transpose(0, 1).matmul(t1)
            t1 = x.matmul(t0)
            return t0.sum() + t1.sum()

        x = torch.randn((3, 2), device=device)
        y = torch.randn((2, 4), device=device)
        z = torch.randn((2, 3), device=device)
        return (f, (x, y, z))

    def test_python_wrapper(self):
        f, args = self._generate(device="cuda")
        compiled = torch.compile(f, dynamic=True)
        result, code = run_and_get_cpp_code(compiled, *args)

        FileCheck().check(
            "pool1 = empty_strided(((4*s0*s1) + (align(4*(s0*s0))), ), (1, )"
        ).check_next(
            "buf0 = alloc_from_pool(pool1, 0, torch.float32, (s0, s0), (s0, 1))"
        ).check(
            "buf1 = alloc_from_pool(pool1, align((4*s0) + (4*s0*((-1) + s0))),"
        ).run(
            code
        )
        self.assertTrue(same(f(*args), result))

    @skipIfRocm
    def test_cpp_wrapper(self):
        f, args = self._generate(device="cuda")
        compiled = torch.compile(f, dynamic=True)
        with config.patch("cpp_wrapper", True):
            result, code = run_and_get_cpp_code(compiled, *args)

        FileCheck().check(
            "auto pool1 = at::empty_strided({(4L*s0*s1) + (align(4L*(static_cast<long>(s0*s0)))), }, {1L, }"
        ).check_next(
            "auto buf0 = alloc_from_pool(pool1, 0, at::kFloat, {s0, s0}, {s0, 1L});"
        ).check(
            "auto buf1 = alloc_from_pool(pool1, align((4*s0) + (4*s0*((-1) + s0))),"
        ).run(
            code
        )
        self.assertTrue(same(f(*args), result))

    @skipIfRocm(msg="test_aot_inductor doesn't work on ROCm")
    def test_abi_compatible(self):
        AOTInductorModelRunner = load_test_module(
            __file__, "inductor.test_aot_inductor"
        ).AOTInductorModelRunner
        f, args = self._generate(device="cuda")
        constraints: List[torch.export.Constraint] = [
            torch._export.dynamic_dim(args[0], 0) >= 1,
            torch._export.dynamic_dim(args[0], 0) <= 2048,
        ]
        with config.patch("aot_inductor.abi_compatible", True):
            result, code = run_and_get_cpp_code(
                lambda: AOTInductorModelRunner.run(
                    "cuda", f, args, constraints=constraints
                )
            )

        FileCheck().check(
            "int64_t int_array_2[] = {24L + (align(12L*s0)), };"
        ).check_next("int64_t int_array_3[] = {1L, };").check_next(
            "AtenTensorHandle pool1_handle;"
        ).check_next(
            "aoti_torch_empty_strided(1, int_array_2, int_array_3,"
        ).check_next(
            "RAIIAtenTensorHandle pool1(pool1_handle);"
        ).check_next(
            "int64_t int_array_4[] = {s0, 3L};"
        ).check_next(
            "int64_t int_array_5[] = {3L, 1L};"
        ).check_next(
            "AtenTensorHandle tmp_tensor_handle_1;"
        ).check_next(
            "aoti_torch__alloc_from_pool(pool1, 0"
        ).run(
            code
        )
        self.assertTrue(same(f(*args), result))


if __name__ == "__main__":
    from torch.testing._internal.inductor_utils import run_inductor_tests

    run_inductor_tests(triton=True)
