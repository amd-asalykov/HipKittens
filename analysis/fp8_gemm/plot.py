import json
import matplotlib.pyplot as plt
import numpy as np

colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#DE836B", "#55555A"]

# Color mapping to match BF16 plot
color_map = {
    "hipblaslt": "#E59952",      # Orange (same as BF16)
    "hipkittens": "#7CB9BC",     # Light Blue/Teal (same as BF16) 
    "composable_kernel": "#DE836B"  # Coral/Light Red (same as BF16)
}


mi355x_baselines = {
    "ck": {
        "1024": 287.617,
        "2048": 455.792,
        "4096": 549.589,
        "8192": "OOM",
        "16384": "OOM",
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


for device in ['mi350x', 'mi355x']:

    # Read data
    try:
        with open(f'mi350x/{device}_fp8_gemm.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading mi350x/{device}_fp8_gemm.json: {e}")
        continue

    # Extract data for plotting
    matrix_sizes = sorted([int(size) for size in data.keys()])
    tk_tflops = [data[str(size)]['tflops'] for size in matrix_sizes]
    hipblaslt_tflops = [data[str(size)]['tflops_hipblaslt'] for size in matrix_sizes]

    ck_tflops = []
    if device == 'mi355x':
        ck_tflops = [mi355x_baselines['ck'][str(size)] for size in matrix_sizes]

    # Process data to separate OOM values
    hipblaslt_vals, hipblaslt_oom = process_data(hipblaslt_tflops)
    tk_vals, tk_oom = process_data(tk_tflops)
    ck_vals, ck_oom = process_data(ck_tflops) if ck_tflops else ([], [])

    max_tflops = max(max(hipblaslt_vals), max(tk_vals), max(ck_vals))

    # Create bar chart
    x = np.arange(len(matrix_sizes)) * 0.2  # Reduce spacing between clusters
    width = 0.06

    fig, ax = plt.subplots(figsize=(15, 8))
    first_bar = x - 3*width
    second_bar = x - 2*width
    third_bar = x - width
    fourth_bar = x
    fifth_bar = x + width
    sixth_bar = x + 2*width
    bars2 = ax.bar(second_bar, hipblaslt_vals, width, label='HipblasLT', color=color_map["hipblaslt"])
    bars3 = ax.bar(third_bar, tk_vals, width, label='HipKittens', color=color_map["hipkittens"])
    bars5 = ax.bar(first_bar, ck_vals, width, label='Composable Kernel', color=color_map["composable_kernel"])

    # Plot X markers for OOM
    oom_height = 150

    for idx in hipblaslt_oom:
        ax.plot(x[idx] - 2*width, oom_height, 'x', color=color_map["hipblaslt"], markersize=25, markeredgewidth=6)
        ax.text(x[idx] - 2*width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=18, color=color_map["hipblaslt"])

    for idx in tk_oom:
        ax.plot(x[idx], oom_height, 'x', color=color_map["hipkittens"], markersize=25, markeredgewidth=6)
        ax.text(x[idx], oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=18, color=color_map["hipkittens"])

    for idx in ck_oom:
        ax.plot(x[idx] - 3*width, oom_height, 'x', color=color_map["composable_kernel"], markersize=25, markeredgewidth=6)
        ax.text(x[idx] - 3*width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=18, color=color_map["composable_kernel"])

    # Add value labels on bars
    for bar, value in zip(bars2, hipblaslt_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=20)

    for bar, value in zip(bars3, tk_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=20)

    for bar, value in zip(bars5, ck_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=20)

    # Set y-axis with more spaced out ticks
    ax.set_ylim(0, max_tflops * 1.15)
    # Create more spaced out y-axis ticks (every 200 TFLOPS instead of default)
    ax.set_xlabel('Matrix Size (N×N)', fontsize=26)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=26)
    ax.set_title(f'FP8 GEMM Performance Comparison {device.upper()}', fontsize=26)
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(fontsize=24)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_fp8_gemm_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Matrix sizes tested: {matrix_sizes}")
    print(f"HipblasLT TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in hipblaslt_vals]}")
    print(f"HipKittens TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in tk_vals]}")
    print(f"Composable Kernel TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in ck_vals]}")