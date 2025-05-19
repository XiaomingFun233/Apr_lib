import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import apr
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F


def reference_flash(q, k, v, mask=None):
    """
    Compute attention using PyTorch's scaled dot product attention.
    
    Args:
        q (Tensor): Query tensor of shape (..., L, E)
        k (Tensor): Key tensor of shape (..., S, E)
        v (Tensor): Value tensor of shape (..., S, Ev)
        mask (Tensor, optional): Attention mask of shape (..., L, S)
    
    Returns:
        Tensor: Attention output of shape (..., L, Ev)
    """
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class Testflash(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)


        return [
            [make_tensor(3), make_tensor(3), 1],
            [make_tensor(20), make_tensor(20), 3.14],
            [make_tensor(20), make_nondiff_tensor(20), -123],
            [make_nondiff_tensor(2, 3), make_tensor(2, 3), -0.3],
        ]

    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            result = ops.apr.flash(*args)
            expected = reference_flash(*args)
            torch.testing.assert_close(result, expected)

    def test_correctness_cpu(self):
        self._test_correctness("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        self._test_correctness("cuda")

    def _opcheck(self, device):
        # Use opcheck to check for incorrect usage of operator registration APIs
        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            opcheck(torch.ops.apr.flash.default, args)

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")




if __name__ == "__main__":
    unittest.main()