import json
import matplotlib.pyplot as plt
import numpy as np

colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#D84A4A", "#4A8ED8", "#FF6B9D"]


mi355x_baselines = {
    "triton": {
        "1024": 130.300560,
        "2048": 599,
        "4096": 993.69,
        "8192": 1143.91,
        "16384": 1087.09,
    },
    # "rocblas": {
    #     "1024": 158.84,
    #     "2048": 318,
    #     "4096": 1201,
    #     "8192": 1303,
    #     "16384": 1301,
    # },
    "hipblaslt": {
        "1024": 165.829,
        "2048": 598.810,
        "4096": 1111.09,
        "8192": 1379.180,
        "16384": 1335.310,
    },
    "ck": {
        "1024": 170.212,
        "2048": 252.214,
        "4096": 954.717,
        "8192": 963.052,
        "16384": 500.964,
    }
}

mi350x_baselines = {
    "triton": {
        "1024": 0,
        "2048": 0,
        "4096": 0,
        "8192": 0,
        "16384": 0,
    },
    "hipblaslt": {
        "1024": 0,
        "2048": 0,
        "4096": 0,
        "8192": 0,
        "16384": 0,
    },
    # "rocblas": {
    #     "1024": 0,
    #     "2048": 0,
    #     "4096": 0,
    #     "8192": 0,
    #     "16384": 0,
    # },
    "ck": {
        "1024": 180.422,
        "2048": 236.799,
        "4096": 770.77,
        "8192": 760.005,
        "16384": 370.43,
    }
}


def process_data(data_list):
    """Separate numeric values and OOM indices"""
    values = []
    oom_indices = []
    for i, val in enumerate(data_list):
        if val == "OOM":
            values.append(0)
            oom_indices.append(i)
        else:
            values.append(val)
    return values, oom_indices


for device in ['mi300x', 'mi325x', 'mi350x', 'mi355x']:

    # Read data
    try:
        with open(f'mi350x/{device}_bf16_gemm.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading mi350x/{device}_bf16_gemm.json: {e}")
        continue

    # Extract data for plotting
    matrix_sizes = sorted([int(size) for size in data.keys()])
    pytorch_tflops = [data[str(size)]['tflops_pytorch'] for size in matrix_sizes]
    aiter_tflops = [data[str(size)]['tflops_aiter'] for size in matrix_sizes]
    tk_tflops = [data[str(size)]['tflops'] for size in matrix_sizes]

    triton_tflops = []
    # rocblas_tflops = []
    ck_tflops = []
    hipblaslt_tflops = []
    if device == 'mi355x':
        triton_tflops = [mi355x_baselines['triton'][str(size)] for size in matrix_sizes]
        # rocblas_tflops = [mi355x_baselines['rocblas'][str(size)] for size in matrix_sizes]
        ck_tflops = [mi355x_baselines['ck'][str(size)] for size in matrix_sizes]
        hipblaslt_tflops = [mi355x_baselines['hipblaslt'][str(size)] for size in matrix_sizes]
    elif device == 'mi350x':
        triton_tflops = [mi350x_baselines['triton'][str(size)] for size in matrix_sizes]
        ck_tflops = [mi350x_baselines['ck'][str(size)] for size in matrix_sizes]
        hipblaslt_tflops = [mi350x_baselines['hipblaslt'][str(size)] for size in matrix_sizes]
        # rocblas_tflops = [mi350x_baselines['rocblas'][str(size)] for size in matrix_sizes]

    # Process data to separate OOM values
    pytorch_vals, pytorch_oom = process_data(pytorch_tflops)
    aiter_vals, aiter_oom = process_data(aiter_tflops)
    hipblaslt_vals, hipblaslt_oom = process_data(hipblaslt_tflops)
    tk_vals, tk_oom = process_data(tk_tflops)
    triton_vals, triton_oom = process_data(triton_tflops) if triton_tflops else ([], [])
    # rocblas_vals, rocblas_oom = process_data(rocblas_tflops) if rocblas_tflops else ([], [])
    ck_vals, ck_oom = process_data(ck_tflops) if ck_tflops else ([], [])

    max_tflops = max(max(pytorch_vals), max(aiter_vals), max(hipblaslt_vals), max(tk_vals), max(triton_vals), max(ck_vals))

    # Create bar chart
    x = np.arange(len(matrix_sizes))
    width = 0.12

    fig, ax = plt.subplots(figsize=(12, 6))
    bars0 = ax.bar(x - 3*width, pytorch_vals, width, label='PyTorch', color=colors[0])
    bars1 = ax.bar(x - 2*width, aiter_vals, width, label='AITER (AMD)', color=colors[1])
    bars2 = ax.bar(x - width, hipblaslt_vals, width, label='HipblasLT', color=colors[2])
    bars3 = ax.bar(x, tk_vals, width, label='HipKittens', color=colors[3])
    bars4 = ax.bar(x + width, triton_vals, width, label='Triton', color=colors[4])
    # bars5 = ax.bar(x + 2*width, rocblas_vals, width, label='rocBLAS', color=colors[5])
    bars6 = ax.bar(x + 3*width, ck_vals, width, label='Composable Kernel', color=colors[6])

    # Plot X markers for OOM
    oom_height = max_tflops * 0.95

    for idx in pytorch_oom:
        ax.plot(x[idx] - 3*width, oom_height, 'x', color=colors[0], markersize=15, markeredgewidth=3)
        ax.text(x[idx] - 3*width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=10, color=colors[0])

    for idx in aiter_oom:
        ax.plot(x[idx] - 2*width, oom_height, 'x', color=colors[1], markersize=15, markeredgewidth=3)
        ax.text(x[idx] - 2*width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=10, color=colors[1])

    for idx in hipblaslt_oom:
        ax.plot(x[idx] - width, oom_height, 'x', color=colors[2], markersize=15, markeredgewidth=3)
        ax.text(x[idx] - width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=10, color=colors[2])

    for idx in tk_oom:
        ax.plot(x[idx], oom_height, 'x', color=colors[3], markersize=15, markeredgewidth=3)
        ax.text(x[idx], oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=10, color=colors[3])

    for idx in triton_oom:
        ax.plot(x[idx] + width, oom_height, 'x', color=colors[4], markersize=15, markeredgewidth=3)
        ax.text(x[idx] + width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=10, color=colors[4])

    # for idx in rocblas_oom:
    #     ax.plot(x[idx] + 2*width, oom_height, 'x', color=colors[5], markersize=15, markeredgewidth=3)
    #     ax.text(x[idx] + 2*width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=10, color=colors[5])

    for idx in ck_oom:
        ax.plot(x[idx] + 3*width, oom_height, 'x', color=colors[6], markersize=15, markeredgewidth=3)
        ax.text(x[idx] + 3*width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=10, color=colors[6])

    # Add value labels on bars
    for bar, value in zip(bars0, pytorch_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    for bar, value in zip(bars1, aiter_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    for bar, value in zip(bars2, hipblaslt_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    for bar, value in zip(bars3, tk_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    for bar, value in zip(bars4, triton_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    # for bar, value in zip(bars5, rocblas_vals):
    #     if value > 0:
    #         height = bar.get_height()
    #         ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
    #                 f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    for bar, value in zip(bars6, ck_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    # add some padding to the top of the y-axis to prevent label overlap
    ax.set_ylim(0, max_tflops * 1.15)
    ax.set_xlabel('Matrix Size (N×N)', fontsize=16)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
    ax.set_title(f'BF16 GEMM Performance Comparison {device.upper()}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_bf16_gemm_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Matrix sizes tested: {matrix_sizes}")
    print(f"PyTorch TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in pytorch_vals]}")
    print(f"AITER (AMD) TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in aiter_vals]}")
    print(f"HipblasLT TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in hipblaslt_vals]}")
    print(f"HipKittens TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in tk_vals]}")
    print(f"Triton TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in triton_vals]}")
    # print(f"rocBLAS TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in rocblas_vals]}")
    print(f"Composable Kernel TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in ck_vals]}")