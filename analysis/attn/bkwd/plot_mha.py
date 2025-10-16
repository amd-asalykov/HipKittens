import json
import matplotlib.pyplot as plt
import numpy as np

colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC"]

for device in ['mi350x']:

    # Read data
    try:
        with open(f'{device}_data_to_log.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {device}_data_to_log.json: {e}")
        continue

    # Extract MHA_bkwd_asm_interleaved data
    mha_data = {}
    for key, value in data.items():
        if key.startswith('MHA_bkwd_asm_interleaved_'):
            n_value = value['N']
            mha_data[n_value] = {
                'tflops_tk': value['tflops_tk'],
                'tflops_ref': value['tflops_ref']
            }

    # Sort by N value
    n_values = sorted(mha_data.keys())
    tk_tflops = [mha_data[n]['tflops_tk'] for n in n_values]
    ref_tflops = [mha_data[n]['tflops_ref'] for n in n_values]

    # Create grouped bar chart
    x = np.arange(len(n_values))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, ref_tflops, width, label='AITER (AMD)', color=colors[1])
    bars2 = ax.bar(x, tk_tflops, width, label='HipKittens', color=colors[3])

    max_tflops = max(max(tk_tflops), max(ref_tflops))

    # Add value labels on bars
    for bar, value in zip(bars1, ref_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    for bar, value in zip(bars2, tk_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    # add some padding to the top of the y-axis to prevent label overlap
    ax.set_ylim(0, max_tflops * 1.15)
    ax.set_xlabel('Sequence Length (N)', fontsize=16)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
    ax.set_title(f'MHA Backwards Performance Comparison {device.upper()}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_mha_bkwd_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Sequence lengths tested: {n_values}")
    print(f"AITER (ASM) TFLOPS: {[f'{t:.2f}' for t in ref_tflops]}")
    print(f"HipKittens TFLOPS: {[f'{t:.2f}' for t in tk_tflops]}")
