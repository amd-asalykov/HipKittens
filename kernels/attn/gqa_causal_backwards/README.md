## Commands

```bash
make SRC=attn_bkwd_causal.cpp TARGET=tk_kernel_bkwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096

make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096

make SRC=attn_fwd_causal.cpp TARGET=tk_kernel_fwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096

python test_python.py 16 4096 64 8 1 
```