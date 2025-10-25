#!/usr/bin/env python3
"""
Phase Solver: Systematically test which threads are in the same phase
by checking for bank conflicts when accessing the same bank.
"""

import torch
import subprocess
import pandas as pd
import os

def run_profiling(should_I_write, offset):
    """Run the kernel with profiling and return conflict information."""

    # Save current test configuration
    test_code = f"""import torch
import tk_kernel

should_I_write = torch.tensor({should_I_write.cpu().tolist()}, dtype=torch.uint32, device='cuda')
offset = torch.tensor({offset.cpu().tolist()}, dtype=torch.uint32, device='cuda')

tk_kernel.dispatch_micro(should_I_write, offset)
"""

    with open('_temp_test.py', 'w') as f:
        f.write(test_code)


    # Run with profiling
    cmd = [
        'rocprofv3',
        '--pmc', 'SQ_INSTS_LDS', 'SQ_LDS_BANK_CONFLICT',
        '--output-format', 'csv',
        '--output-file', 'phase_test',
        '-d', 'out',
        '--',
        'python3', '_temp_test.py'
    ]

    try:
        # Suppress output to keep things clean
        result = subprocess.run(cmd, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Check if command succeeded
        if result.returncode != 0:
            print(f"\nrocprofv3 failed with return code {result.returncode}")
            return None

    except subprocess.TimeoutExpired:
        print(f"Timeout!")
        return None

    # Read results
    try:
        df = pd.read_csv("out/phase_test_counter_collection.csv")
        # Keep only LDS-related counters
        df = df[df["Counter_Name"].isin(["SQ_INSTS_LDS", "SQ_LDS_BANK_CONFLICT"])]

        # Pivot so each dispatch becomes one row with both counters
        pivot = df.pivot_table(index=["Dispatch_Id", "Kernel_Name"],
                            columns="Counter_Name",
                            values="Counter_Value",
                            aggfunc="first").reset_index()

        # Get the counts from the most recent dispatch (last row)
        last_dispatch = pivot.iloc[-1]
        conflicts = last_dispatch["SQ_LDS_BANK_CONFLICT"]
        lds_insts = last_dispatch["SQ_INSTS_LDS"]
        conflict_ratio = conflicts / lds_insts if lds_insts > 0 else 0

        return lds_insts, conflicts, conflict_ratio
    except Exception as e:
        print(f"\nError reading profiling data: {e}")
        return None
    finally:
        # Cleanup temp file
        if os.path.exists('_temp_test.py'):
            os.remove('_temp_test.py')


def test_phase(thread_0, thread_i):
    """Test if two threads are in the same phase by checking for bank conflicts."""

    # Create should_I_write: only thread_0 and thread_i are enabled
    should_I_write = torch.zeros(64, dtype=torch.uint32, device='cuda')
    should_I_write[thread_0] = 1
    should_I_write[thread_i] = 1

    # Each thread writes to offset tid * 64 * 4 (all map to same bank)
    offset = torch.zeros(64, dtype=torch.uint32, device='cuda')
    for tid in range(64):
        offset[tid] = tid * 64 * 4

    result = run_profiling(should_I_write, offset)

    if result is None:
        return None

    lds_insts, conflicts, conflict_ratio = result

    # If there's a conflict, threads are in the same phase
    has_conflict = conflicts > 0

    return has_conflict


def solve_phases():
    """Solve for the phase assignment of all threads by testing all pairs."""

    print("=" * 60)
    print("Phase Solver: Testing all thread pairs for conflicts")
    print("=" * 60)
    print()

    NUM_THREADS = 64

    # Build conflict matrix: conflict_matrix[i][j] = True if threads i and j conflict
    conflict_matrix = [[False] * NUM_THREADS for _ in range(NUM_THREADS)]

    # Test all pairs
    total_tests = (NUM_THREADS * (NUM_THREADS - 1)) // 2
    test_count = 0

    for thread_i in range(NUM_THREADS):
        for thread_j in range(thread_i + 1, NUM_THREADS):
            test_count += 1
            print(f"[{test_count}/{total_tests}] Testing thread {thread_i} vs thread {thread_j}... ", end='', flush=True)

            has_conflict = test_phase(thread_i, thread_j)

            if has_conflict is None:
                print("ERROR")
                continue

            conflict_matrix[thread_i][thread_j] = has_conflict
            conflict_matrix[thread_j][thread_i] = has_conflict

            if has_conflict:
                print("CONFLICT")
            else:
                print("no conflict")

    print()
    print("=" * 60)
    print("Grouping threads by conflict patterns")
    print("=" * 60)
    print()

    # Group threads that all conflict with each other into the same phase
    phases = {}  # thread_id -> phase_id
    phase_groups = {}  # phase_id -> list of thread_ids
    unassigned = list(range(NUM_THREADS))
    current_phase = 0

    while unassigned:
        # Start a new phase with the first unassigned thread
        representative = unassigned[0]
        phase_groups[current_phase] = [representative]
        phases[representative] = current_phase
        unassigned.remove(representative)

        # Find all threads that conflict with the representative
        threads_to_remove = []
        for thread_i in unassigned:
            if conflict_matrix[representative][thread_i]:
                phases[thread_i] = current_phase
                phase_groups[current_phase].append(thread_i)
                threads_to_remove.append(thread_i)

        # Remove assigned threads
        for thread in threads_to_remove:
            unassigned.remove(thread)

        print(f"Phase {current_phase}: {len(phase_groups[current_phase])} threads - {phase_groups[current_phase]}")
        current_phase += 1

    print()
    print("=" * 60)
    print("Phase Assignment Results")
    print("=" * 60)
    print()

    # Print summary
    for phase_id in sorted(phase_groups.keys()):
        threads = phase_groups[phase_id]
        print(f"Phase {phase_id}: {len(threads)} threads - {threads}")

    print()
    print(f"Total phases detected: {len(phase_groups)}")
    print()

    # Save results
    with open('phase_results.txt', 'w') as f:
        f.write("Phase Assignment Results\n")
        f.write("=" * 60 + "\n\n")
        for phase_id in sorted(phase_groups.keys()):
            threads = phase_groups[phase_id]
            f.write(f"Phase {phase_id}: {len(threads)} threads - {threads}\n")
        f.write(f"\nTotal phases: {len(phase_groups)}\n")

        # Save conflict matrix
        f.write("\n" + "=" * 60 + "\n")
        f.write("Conflict Matrix (1 = conflict, 0 = no conflict)\n")
        f.write("=" * 60 + "\n\n")
        f.write("   ")
        for j in range(NUM_THREADS):
            f.write(f"{j:2d} ")
        f.write("\n")
        for i in range(NUM_THREADS):
            f.write(f"{i:2d} ")
            for j in range(NUM_THREADS):
                f.write(f" {'1' if conflict_matrix[i][j] else '0'} ")
            f.write("\n")

    print("Results saved to phase_results.txt")

    return phases, phase_groups, conflict_matrix


if __name__ == '__main__':
    phases, phase_groups, conflict_matrix = solve_phases()
