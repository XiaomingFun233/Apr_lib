import torch
from torch import Tensor

__all__ = ["flash"]
'''
const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V,
                 int n, int m, int d_k, int d_v
'''
def flash(Q: Tensor, K: Tensor, V: Tensor, n: int, m: int, d_k: int, d_v: int) -> Tensor:
    """Performs FLASH ATTENTION in an efficient fused kernel"""
    return torch.ops.apr.flash.default(Q, K, V, n, m, d_k, d_v)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("apr::flash")
def _(Q, K, V, n, m, d_k, d_v):
    torch._check(Q.shape == K.shape)
    torch._check(K.shape == V.shape)
    torch._check(Q.dtype == torch.float)
    torch._check(K.dtype == torch.float)
    torch._check(V.dtype == torch.float)
    torch._check(Q.device == K.device and K.device == V.device)
    return torch.empty_like(Q)




