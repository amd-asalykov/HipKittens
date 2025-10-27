import torch
import tk_kernel
import tk_kernel_asm
import random
import time
import math
from torch.nn.functional import scaled_dot_product_attention
import aiter

torch.set_printoptions(
    precision=3,        # decimals
    sci_mode=False,     # True → always scientific
    linewidth=220,      # characters per line before folding
    threshold=float("inf")  # print every element, no summarising "..."
)

torch.cuda.set_device(6)

# Inputs
N = 128
M = 64
K = 64
dtype = torch.bfloat16

def robustness_check(ref, pred):
    ref = ref.float()
    pred = pred.float()
    diff = (ref - pred).abs()
    denom = ref.abs().clamp_min(1e-6)
    mask = (diff > (0.001 + 0.05 * denom))
    error_count = mask.sum().item()
    numel = ref.numel()
    rel_error = error_count / numel
    l2_error = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), pred.flatten(), dim=0).item()
    return diff, error_count, numel, rel_error, l2_error, cos, mask  

# AITER
torch.manual_seed(0)
random.seed(0)
A = torch.randn(K, N, dtype=dtype, device='cuda', requires_grad=True)
B = torch.randn(K, M, dtype=dtype, device='cuda', requires_grad=True)

C_ref = torch.matmul(A.transpose(-1, -2), B).transpose(-1, -2).float()

A_hk = A.clone()
B_hk = B.clone()
C_hk = torch.zeros(M, N, dtype=torch.float32, device='cuda', requires_grad=True)
torch.cuda.synchronize()
tk_kernel.dispatch_micro(A_hk, B_hk, C_hk)
torch.cuda.synchronize()

A_hk_asm = A.clone()
B_hk_asm = B.clone()
C_hk_asm = torch.zeros(M, N, dtype=torch.float32, device='cuda', requires_grad=True)
torch.cuda.synchronize()
tk_kernel_asm.dispatch_micro(A_hk_asm, B_hk_asm, C_hk_asm)
torch.cuda.synchronize()

print(f"\nHK vs Reference comparison:")
print("C_ref: ", C_ref[0, :16], "Max:", C_ref.max().item())
print("C_hk: ", C_hk[0, :16], "Max:", C_hk.max().item())
print("C_hk_asm: ", C_hk_asm[0, :16], "Max:", C_hk_asm.max().item())

print("Robustness check:")
c_diff, c_err_cnt, c_total, c_rel_error, c_l2_error, c_cos, c_mask = robustness_check(C_ref, C_hk)
print(f"C: max_abs={c_diff.max().item():.6f}, max_rel={c_rel_error:.4f}, "
      f"rel_l2={c_l2_error:.4f}, cos={c_cos:.6f}, "
      f"errors={c_err_cnt}/{c_total} ({100*c_err_cnt/c_total:.4f}%)")

c_diff, c_err_cnt, c_total, c_rel_error, c_l2_error, c_cos, c_mask = robustness_check(C_ref, C_hk_asm)
print(f"C_asm: max_abs={c_diff.max().item():.6f}, max_rel={c_rel_error:.4f}, "
      f"rel_l2={c_l2_error:.4f}, cos={c_cos:.6f}, "
      f"errors={c_err_cnt}/{c_total} ({100*c_err_cnt/c_total:.4f}%)")

c_diff, c_err_cnt, c_total, c_rel_error, c_l2_error, c_cos, c_mask = robustness_check(C_hk, C_hk_asm)
print(f"C_hk vs C_hk_asm: max_abs={c_diff.max().item():.6f}, max_rel={c_rel_error:.4f}, "
      f"rel_l2={c_l2_error:.4f}, cos={c_cos:.6f}, "
      f"errors={c_err_cnt}/{c_total} ({100*c_err_cnt/c_total:.4f}%)")