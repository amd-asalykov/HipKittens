#!/usr/bin/env python3
"""
Test a single phase comparison to verify profiling works
"""

import torch
import tk_kernel

print("Testing threads 0 and 1 writing to same bank...")

# Only threads 0 and 1 enabled
should_I_write = torch.zeros(64, dtype=torch.uint32, device='cuda')
should_I_write[0] = 1
should_I_write[1] = 1

# Each thread writes to offset tid * 64 * 4 (all map to same bank)
offset = torch.zeros(64, dtype=torch.uint32, device='cuda')
for tid in range(64):
    offset[tid] = tid * 64 * 4

print(f"Thread 0 writes to offset: {offset[0].item()}")
print(f"Thread 1 writes to offset: {offset[1].item()}")

tk_kernel.dispatch_micro(should_I_write, offset)

print("Done! Now check for conflicts with:")
print("  python3 out/analyze_conflicts.py")
