DEVICE=0

# GQA

make TARGET=tk_kernel SRC=attn_fwd_non_causal_d64.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=1024 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 16 1024 64 8 0 mi355x_d64_gqa_non_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_non_causal_d64.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=2048 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 16 2048 64 8 0 mi355x_d64_gqa_non_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_non_causal_d64.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 16 4096 64 8 0 mi355x_d64_gqa_non_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_non_causal_d64.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=8192 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 16 8192 64 8 0 mi355x_d64_gqa_non_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_non_causal_d64.cpp ATTN_B=15 ATTN_H=64 ATTN_H_KV=8 ATTN_N=16384 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 15 16384 64 8 0 mi355x_d64_gqa_non_causal_fwd.json 64

# MHA

make TARGET=tk_kernel SRC=attn_fwd_non_causal_d64.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=1024 ATTN_D=64
python test_python.py 16 1024 16 16 0 mi355x_d64_mha_non_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_non_causal_d64.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=2048 ATTN_D=64
python test_python.py 16 2048 16 16 0 mi355x_d64_mha_non_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_non_causal_d64.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=4096 ATTN_D=64
python test_python.py 16 4096 16 16 0 mi355x_d64_mha_non_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_non_causal_d64.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=8192 ATTN_D=64
python test_python.py 16 8192 16 16 0 mi355x_d64_mha_non_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_non_causal_d64.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=16384 ATTN_D=64
python test_python.py 16 16384 16 16 0 mi355x_d64_mha_non_causal_fwd.json 64


# CAUSAL
# GQA

make TARGET=tk_kernel SRC=attn_fwd_causal_d64.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=1024 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 16 1024 64 8 1 mi355x_d64_gqa_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_causal_d64.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=2048 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 16 2048 64 8 1 mi355x_d64_gqa_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_causal_d64.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 16 4096 64 8 1 mi355x_d64_gqa_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_causal_d64.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=8192 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 16 8192 64 8 1 mi355x_d64_gqa_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_causal_d64.cpp ATTN_B=15 ATTN_H=64 ATTN_H_KV=8 ATTN_N=16384 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 15 16384 64 8 1 mi355x_d64_gqa_causal_fwd.json 64

# MHA

make TARGET=tk_kernel SRC=attn_fwd_causal_d64.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=1024 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 16 1024 16 16 1 mi355x_d64_mha_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_causal_d64.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=2048 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 16 2048 16 16 1 mi355x_d64_mha_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_causal_d64.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=4096 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 16 4096 16 16 1 mi355x_d64_mha_causal_fwd.json 64 

make TARGET=tk_kernel SRC=attn_fwd_causal_d64.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=8192 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 16 8192 16 16 1 mi355x_d64_mha_causal_fwd.json 64

make TARGET=tk_kernel SRC=attn_fwd_causal_d64.cpp ATTN_B=15 ATTN_H=16 ATTN_H_KV=16 ATTN_N=16384 ATTN_D=64
CUDA_VISIBLE_DEVICES=$DEVICE python test_python.py 15 16384 16 16 1 mi355x_d64_mha_causal_fwd.json 64

