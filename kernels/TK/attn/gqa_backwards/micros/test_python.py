import torch
import random
import math
import tk_kernel

torch.set_printoptions(
    precision=3,        
    sci_mode=False,     
    linewidth=220,      
    threshold=float("inf")  
)


causal = False
b = 1
h = 1
n = 32
d = 32
mean = 5 
std = 0.1
dtype = torch.bfloat16

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    return scaled_tensor.contiguous()

# pytorch
x = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
vec = generate_tensor((b, h, n, 1), mean, std, torch.bfloat16, 'cuda')
y = x.sum(dim=-1, keepdim=True) 
# y = x * vec
y = vec

# tk
y_tk = torch.zeros_like(y)
tk_kernel.dispatch_micro(x, vec, y_tk)

# check
diff = (y - y_tk).abs().max()
print(f"diff: {diff}")

# print(y[0, 0, 3:6, :16])
# print(y_tk[0, 0, 3:6, :16])

print(y[0, 0, :].T)
print(y_tk[0, 0, :].T)