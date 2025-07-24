import torch
import tk_kernel

A = torch.randn(1, 1, 1024, 1024, dtype=torch.bfloat16, device='cuda')

tk_kernel.dispatch_micro(A)