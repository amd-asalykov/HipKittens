## Commands

```bash
make SRC=attn_bkwd_non_causal.cpp TARGET=tk_kernel_bkwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=2048

make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=2048

make SRC=attn_fwd_non_causal.cpp TARGET=tk_kernel_fwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=2048

python test_python.py 16 2048 64 8 0 
```

