
# GQA

make TARGET=tk_kernel SRC=attn_fwd_non_causal.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=1024
python test_python.py 16 1024 64 8 0 mi355x_gqa_non_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_non_causal.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=2048
python test_python.py 16 2048 64 8 0 mi355x_gqa_non_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_non_causal.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096
python test_python.py 16 4096 64 8 0 mi355x_gqa_non_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_non_causal.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=8192
python test_python.py 16 8192 64 8 0 mi355x_gqa_non_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_non_causal.cpp ATTN_B=15 ATTN_H=64 ATTN_H_KV=8 ATTN_N=16384
python test_python.py 15 16384 64 8 0 mi355x_gqa_non_causal_fwd.json

# MHA

make TARGET=tk_kernel SRC=attn_fwd_non_causal.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=1024
python test_python.py 16 1024 16 16 0 mi355x_mha_non_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_non_causal.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=2048
python test_python.py 16 2048 16 16 0 mi355x_mha_non_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_non_causal.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=4096
python test_python.py 16 4096 16 16 0 mi355x_mha_non_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_non_causal.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=8192
python test_python.py 16 8192 16 16 0 mi355x_mha_non_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_non_causal.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=16384
python test_python.py 16 16384 16 16 0 mi355x_mha_non_causal_fwd.json


## CAUSAL
# GQA

make TARGET=tk_kernel SRC=attn_fwd_causal.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=1024
python test_python.py 16 1024 64 8 1 mi355x_gqa_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_causal.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=2048
python test_python.py 16 2048 64 8 1 mi355x_gqa_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_causal.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096
python test_python.py 16 4096 64 8 1 mi355x_gqa_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_causal.cpp ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=8192
python test_python.py 16 8192 64 8 1 mi355x_gqa_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_causal.cpp ATTN_B=15 ATTN_H=64 ATTN_H_KV=8 ATTN_N=16384
python test_python.py 15 16384 64 8 1 mi355x_gqa_causal_fwd.json

# MHA

make TARGET=tk_kernel SRC=attn_fwd_causal.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=1024
python test_python.py 16 1024 16 16 1 mi355x_mha_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_causal.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=2048
python test_python.py 16 2048 16 16 1 mi355x_mha_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_causal.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=4096
python test_python.py 16 4096 16 16 1 mi355x_mha_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_causal.cpp ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=8192
python test_python.py 16 8192 16 16 1 mi355x_mha_causal_fwd.json

make TARGET=tk_kernel SRC=attn_fwd_causal.cpp ATTN_B=15 ATTN_H=16 ATTN_H_KV=16 ATTN_N=16384
python test_python.py 16 16384 16 16 1 mi355x_mha_causal_fwd.json

