#!/usr/bin/env python3
"""
Quick test to verify the setup works - test just thread 0 vs thread 1
"""

import torch
import tk_kernel

print("Testing basic setup...")
print()

# Test 1: Only thread 0 writes
print("Test 1: Only thread 0 writes to offset 0")
should_I_write = torch.zeros(64, dtype=torch.uint32, device='cuda')
should_I_write[0] = 1
offset = torch.zeros(64, dtype=torch.uint32, device='cuda')
tk_kernel.dispatch_micro(should_I_write, offset)
print("✓ Success - no crash")
print()

# Test 2: Thread 0 and thread 1 both write to offset 0 (same bank)
print("Test 2: Thread 0 and thread 1 both write to offset 0 (same bank)")
should_I_write = torch.zeros(64, dtype=torch.uint32, device='cuda')
should_I_write[0] = 1
should_I_write[1] = 1
offset = torch.zeros(64, dtype=torch.uint32, device='cuda')
tk_kernel.dispatch_micro(should_I_write, offset)
print("✓ Success - no crash")
print()

# Test 3: Thread 0 and thread 1 write to different offsets
print("Test 3: Thread 0 writes to offset 0, thread 1 writes to offset 128")
should_I_write = torch.zeros(64, dtype=torch.uint32, device='cuda')
should_I_write[0] = 1
should_I_write[1] = 1
offset = torch.zeros(64, dtype=torch.uint32, device='cuda')
offset[0] = 0
offset[1] = 128
tk_kernel.dispatch_micro(should_I_write, offset)
print("✓ Success - no crash")
print()

print("All basic tests passed!")
print()
print("Now run the profiling command to see conflict information:")
print("  rocprofv3 --pmc SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT --output-format csv --output-file lds_conflict -d out -- python quick_test.py")
print("  python out/analyze_conflicts.py")
