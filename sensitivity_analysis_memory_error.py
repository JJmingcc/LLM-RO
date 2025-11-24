"""
Sensitivity Analysis for Memory Constraint and Error Rate
Analyzes the impact of varying GPU memory capacity (C_gpu) and error rate thresholds (epsilon)
on the optimization results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from RODIU_LLM import DataGenerator, LLMInferenceOptimizer
import copy
import time
from datetime import datetime
import os

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def run_sensitivity_analysis(C_gpu_scale, psi_error, base_data, time_limit=300, mip_gap=0.01):
    """
    Run optimization with scaled GPU memory capacity and error thresholds.

    Args:
        C_gpu_scale: Scaling factor for GPU memory capacity (C_gpu)
        psi_error: Scaling factor for error rate thresholds (epsilon)
        base_data: Base data object with original thresholds
        time_limit: Optimization time limit in seconds
        mip_gap: MIP gap tolerance

    Returns:
        Dictionary with results including costs, allocations, and metrics
    """
    # Create a copy of data with scaled thresholds
    data = copy.deepcopy(base_data)

    # Scale GPU memory capacity and error thresholds
    data.C_gpu = base_data.C_gpu * C_gpu_scale
    data.epsilon = base_data.epsilon * psi_error

    # Build and solve optimization
    start_time = time.time()
    try:
        optimizer = LLMInferenceOptimizer(data)
        solution = optimizer.build_and_solve_optimization_problem(time_limit=time_limit, mip_gap=mip_gap)
        solve_time = time.time() - start_time

        if solution is None or solution['status'] not in ['OPTIMAL', 'TIME_LIMIT']:
            return {
                'C_gpu_scale': C_gpu_scale,
                'psi_error': psi_error,
                'status': 'INFEASIBLE' if solution is None else solution.get('status', 'INFEASIBLE'),
                'solve_time': solve_time,
                'total_cost': None,
                'gap': None
            }

        # Extract detailed metrics
        results = extract_metrics(optimizer, solution, data, C_gpu_scale, psi_error, solve_time)
        return results

    except Exception as e:
        print(f"Error for C_gpu_scale={C_gpu_scale}, psi_error={psi_error}: {str(e)}")
        return {
            'C_gpu_scale': C_gpu_scale,
            'psi_error': psi_error,
            'status': 'ERROR',
            'error_message': str(e),
            'solve_time': time.time() - start_time,
            'total_cost': None,
            'gap': None
        }

def extract_metrics(optimizer, solution, data, C_gpu_scale, psi_error, solve_time):
    """
    Extract detailed metrics from optimization solution.

    Args:
        optimizer: LLMInferenceOptimizer instance
        solution: Solution dictionary
        data: Data object used for optimization
        C_gpu_scale: Memory capacity scaling factor
        psi_error: Error scaling factor
        solve_time: Time taken to solve

    Returns:
        Dictionary with comprehensive metrics
    """
    d = data

    # Calculate cost components
    C1 = 0  # Resource rental cost
    for (j, k), y_val in solution['y'].items():
        C1 += d.Delta_T * d.p_c[k] * y_val

    C2_model = 0  # Model storage cost
    C2_data = 0   # Data storage cost
    for i in range(d.I):
        for j in range(d.J):
            for k in range(d.K):
                x_val = solution['x'].get((i, j, k), 0)
                if x_val > 0:
                    z_val = optimizer.vars['z'][i, j, k].X
                    C2_model += d.Delta_T * d.p_s * d.B[j] * z_val
                    C2_data += d.Delta_T * d.p_s * d.theta[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x_val
    C2 = C2_model + C2_data

    C3 = optimizer.vars['varrho'].X if hasattr(optimizer.vars['varrho'], 'X') else 0  # Robust delay penalty

    C4 = 0  # Unmet demand penalty
    for i, u_val in solution['u'].items():
        C4 += d.phi[i] * u_val * d.lambda_i[i]

    # Calculate total unmet demand
    total_unmet = sum(solution['u'].values()) if solution['u'] else 0

    # Calculate number of GPUs by type
    num_gpus_by_tier = {}
    total_gpus = 0
    for (j, k), y_val in solution['y'].items():
        tier = d.gpu_tiers[k]
        num_gpus_by_tier[tier] = num_gpus_by_tier.get(tier, 0) + y_val
        total_gpus += y_val

    # Calculate average delay and error rate
    avg_delay = 0
    avg_error = 0
    total_allocation = 0

    for i in range(d.I):
        for j in range(d.J):
            for k in range(d.K):
                x_val = solution['x'].get((i, j, k), 0)
                if x_val > 0:
                    avg_delay += d.d_bar[i, j, k] * (d.h[i] + d.f[i]) * x_val
                    avg_error += d.e_bar[i, j, k] * x_val
                    total_allocation += x_val

    avg_delay = avg_delay / total_allocation if total_allocation > 0 else 0
    avg_error = avg_error / total_allocation if total_allocation > 0 else 0

    # Calculate memory utilization metrics
    avg_memory_util = 0
    max_memory_util = 0
    num_configs = 0

    for j in range(d.J):
        for k in range(d.K):
            if (j, k) in solution['y'] and solution['y'][(j, k)] > 0:
                # Calculate memory usage for this configuration
                memory_used = d.B[j]  # Model weights
                # Add KV cache if applicable
                for i in range(d.I):
                    if solution['x'].get((i, j, k), 0) > 0:
                        memory_used += d.beta[j] * (d.h[i] + d.f[i]) * d.lambda_i[i] * d.T_res[i, j, k] * solution['x'][(i, j, k)]

                memory_capacity = d.C_gpu[k]
                if memory_capacity > 0:
                    util = memory_used / memory_capacity
                    avg_memory_util += util
                    max_memory_util = max(max_memory_util, util)
                    num_configs += 1

    avg_memory_util = avg_memory_util / num_configs if num_configs > 0 else 0

    # Calculate tensor parallelism metrics
    tp_selections = {}
    tp_distribution = {1: 0, 2: 0, 4: 0, 8: 0}
    total_tp_configs = 0

    for (j, k, n), w_val in solution.get('w', {}).items():
        if w_val > 0.5:  # Binary variable is selected
            tp_selections[(j, k)] = n
            tp_distribution[n] += 1
            total_tp_configs += 1

    # Calculate average TP degree (weighted by number of GPUs)
    weighted_tp_sum = 0
    total_gpus_with_tp = 0
    for (j, k), n in tp_selections.items():
        num_gpus = solution['y'].get((j, k), 0)
        weighted_tp_sum += n * num_gpus
        total_gpus_with_tp += num_gpus

    avg_tp_degree = weighted_tp_sum / total_gpus_with_tp if total_gpus_with_tp > 0 else 0

    return {
        'C_gpu_scale': C_gpu_scale,
        'psi_error': psi_error,
        'status': solution['status'],
        'solve_time': solve_time,
        'gap': solution.get('gap', 0),

        # Total cost and components
        'total_cost': solution['objective'],
        'C1_rental': C1,
        'C2_storage': C2,
        'C2_model': C2_model,
        'C2_data': C2_data,
        'C3_robust_delay': C3,
        'C4_unmet': C4,

        # Cost percentages
        'C1_pct': C1 / solution['objective'] * 100,
        'C2_pct': C2 / solution['objective'] * 100,
        'C3_pct': C3 / solution['objective'] * 100,
        'C4_pct': C4 / solution['objective'] * 100,

        # Resource metrics
        'num_allocations': len(solution.get('x', {})),
        'num_gpu_configs': len(solution.get('y', {})),
        'total_gpus': total_gpus,
        'num_unmet_queries': len(solution.get('u', {})),
        'total_unmet_fraction': total_unmet,

        # Performance metrics
        'avg_delay': avg_delay,
        'avg_error': avg_error,
        'avg_memory_util': avg_memory_util,
        'max_memory_util': max_memory_util,

        # Scaled thresholds
        'scaled_C_gpu_mean': np.mean(data.C_gpu),
        'scaled_epsilon_mean': np.mean(data.epsilon),

        # Tensor parallelism metrics
        'num_tp_configs': total_tp_configs,
        'avg_tp_degree': avg_tp_degree,
        'tp_1_count': tp_distribution[1],
        'tp_2_count': tp_distribution[2],
        'tp_4_count': tp_distribution[4],
        'tp_8_count': tp_distribution[8],
    }

def main():
    """Main execution function for sensitivity analysis"""

    print("="*80)
    print("SENSITIVITY ANALYSIS: MEMORY CAPACITY & ERROR RATE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Generate base data
    print("Generating base data using DataGenerator...")
    generator = DataGenerator(seed=42)
    base_data = generator.generate()
    generator.print_summary(base_data)

    # Define scaling factors
    C_gpu_scale_values = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    psi_error_values = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS PARAMETERS")
    print("="*80)
    print(f"Base GPU memory capacity (C_gpu): {base_data.C_gpu}")
    print(f"Base error thresholds (epsilon): {base_data.epsilon}")
    print(f"\nMemory capacity scaling factors (C_gpu_scale): {C_gpu_scale_values}")
    print(f"Error scaling factors (psi_error): {psi_error_values}")
    print(f"Total scenarios: {len(C_gpu_scale_values)} × {len(psi_error_values)} = {len(C_gpu_scale_values) * len(psi_error_values)}")

    # Run sensitivity analysis
    results_list = []
    counter = 0
    total = len(C_gpu_scale_values) * len(psi_error_values)

    print("\n" + "="*80)
    print("Running sensitivity analysis...")
    print("="*80)

    for C_gpu_scale in C_gpu_scale_values:
        for psi_error in psi_error_values:
            counter += 1
            print(f"\n[{counter}/{total}] C_gpu_scale={C_gpu_scale:.1f}, psi_error={psi_error:.1f}")

            result = run_sensitivity_analysis(C_gpu_scale, psi_error, base_data, time_limit=300, mip_gap=0.01)
            results_list.append(result)

            if result['status'] in ['OPTIMAL', 'TIME_LIMIT']:
                print(f"  ✓ {result['status']}: Total Cost = ${result['total_cost']:,.2f}, Gap = {result['gap']:.4f}")
            else:
                print(f"  ✗ {result['status']}")

    # Convert to DataFrame
    df_results = pd.DataFrame(results_list)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Successful optimizations: {len(df_results[df_results['status'].isin(['OPTIMAL', 'TIME_LIMIT'])])} / {total}")

    # Filter successful results for analysis
    df_success = df_results[df_results['status'].isin(['OPTIMAL', 'TIME_LIMIT'])].copy()

    if len(df_success) > 0:
        print(f"\nCost statistics:")
        print(f"  Minimum cost: ${df_success['total_cost'].min():,.2f}")
        print(f"  Maximum cost: ${df_success['total_cost'].max():,.2f}")
        print(f"  Mean cost: ${df_success['total_cost'].mean():,.2f}")
        print(f"  Median cost: ${df_success['total_cost'].median():,.2f}")

        # Save results
        output_dir = 'sensitivity_results'
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = os.path.join(output_dir, f'sensitivity_memory_error_{timestamp}.csv')
        df_results.to_csv(csv_filename, index=False)
        print(f"\nResults saved to: {csv_filename}")

        # Create visualizations
        print("\nGenerating visualizations...")
        create_visualizations(df_success, output_dir, timestamp)

    else:
        print("\n⚠️  WARNING: No successful optimization runs!")
        print("All scenarios failed. Please check the parameter ranges and constraints.")

    print("\n" + "="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    return df_results

def create_visualizations(df, output_dir, timestamp):
    """
    Create comprehensive visualizations for sensitivity analysis results.

    Args:
        df: DataFrame with successful results
        output_dir: Directory to save plots
        timestamp: Timestamp string for filenames
    """

    # Figure 1: Cost Heatmap
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Total Cost Heatmap
    cost_pivot = df.pivot_table(values='total_cost', index='C_gpu_scale', columns='psi_error')
    sns.heatmap(cost_pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0, 0],
                cbar_kws={'label': 'Total Cost ($)'})
    axes[0, 0].set_title('Total Cost Heatmap', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Error Scaling (psi_error)')
    axes[0, 0].set_ylabel('Memory Capacity Scaling (C_gpu)')

    # Rental Cost Heatmap
    rental_pivot = df.pivot_table(values='C1_rental', index='C_gpu_scale', columns='psi_error')
    sns.heatmap(rental_pivot, annot=True, fmt='.0f', cmap='Blues', ax=axes[0, 1],
                cbar_kws={'label': 'Rental Cost ($)'})
    axes[0, 1].set_title('GPU Rental Cost (C1)', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Error Scaling (psi_error)')
    axes[0, 1].set_ylabel('Memory Capacity Scaling (C_gpu)')

    # Unmet Demand Cost Heatmap
    unmet_pivot = df.pivot_table(values='C4_unmet', index='C_gpu_scale', columns='psi_error')
    sns.heatmap(unmet_pivot, annot=True, fmt='.0f', cmap='Reds', ax=axes[1, 0],
                cbar_kws={'label': 'Unmet Demand Cost ($)'})
    axes[1, 0].set_title('Unmet Demand Penalty (C4)', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Error Scaling (psi_error)')
    axes[1, 0].set_ylabel('Memory Capacity Scaling (C_gpu)')

    # Average Memory Utilization Heatmap
    memory_pivot = df.pivot_table(values='avg_memory_util', index='C_gpu_scale', columns='psi_error')
    sns.heatmap(memory_pivot, annot=True, fmt='.2f', cmap='Greens', ax=axes[1, 1],
                cbar_kws={'label': 'Memory Utilization'})
    axes[1, 1].set_title('Average Memory Utilization', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Error Scaling (psi_error)')
    axes[1, 1].set_ylabel('Memory Capacity Scaling (C_gpu)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmaps_memory_error_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: heatmaps_memory_error_{timestamp}.png")
    plt.close()

    # Figure 2: Cost Component Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Line plot for fixed memory scaling
    for C_gpu_scale in sorted(df['C_gpu_scale'].unique()):
        subset = df[df['C_gpu_scale'] == C_gpu_scale]
        axes[0].plot(subset['psi_error'], subset['total_cost'],
                    marker='o', label=f'C_gpu={C_gpu_scale:.1f}×', linewidth=2)

    axes[0].set_xlabel('Error Scaling (psi_error)', fontsize=11)
    axes[0].set_ylabel('Total Cost ($)', fontsize=11)
    axes[0].set_title('Total Cost vs Error Scaling\n(Different Memory Capacities)', fontweight='bold')
    axes[0].legend(title='Memory Scaling', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Line plot for fixed error scaling
    for psi_e in sorted(df['psi_error'].unique())[::2]:  # Show every other error value
        subset = df[df['psi_error'] == psi_e]
        axes[1].plot(subset['C_gpu_scale'], subset['total_cost'],marker='s', label=f'ψ_error={psi_e:.1f}', linewidth=2)

    axes[1].set_xlabel('Memory Capacity Scaling (C_gpu)', fontsize=11)
    axes[1].set_ylabel('Total Cost ($)', fontsize=11)
    axes[1].set_title('Total Cost vs Memory Capacity\n(Different Error Scalings)', fontweight='bold')
    axes[1].legend(title='Error Scaling', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cost_trends_memory_error_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: cost_trends_memory_error_{timestamp}.png")
    plt.close()

    # Figure 3: Cost Component Stacked Bar Chart
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Select representative scenarios
    selected_scenarios = df.sort_values('total_cost').iloc[::len(df)//10][:10]  # 10 representative points

    x_labels = [f"ψ_gpu={row['C_gpu_scale']:.1f}×\nψ_ε={row['psi_error']:.1f}"
                for _, row in selected_scenarios.iterrows()]

    width = 0.6
    x_pos = np.arange(len(selected_scenarios))

    ax.bar(x_pos, selected_scenarios['C1_rental'], width, label='C1: GPU Rental', color='#3498db')
    ax.bar(x_pos, selected_scenarios['C2_storage'], width,
           bottom=selected_scenarios['C1_rental'], label='C2: Storage', color='#2ecc71')
    ax.bar(x_pos, selected_scenarios['C3_robust_delay'], width,
           bottom=selected_scenarios['C1_rental'] + selected_scenarios['C2_storage'],
           label='C3: Robust Delay', color='#f39c12')
    ax.bar(x_pos, selected_scenarios['C4_unmet'], width,
           bottom=selected_scenarios['C1_rental'] + selected_scenarios['C2_storage'] + selected_scenarios['C3_robust_delay'],
           label='C4: Unmet Demand', color='#e74c3c')

    # ax.set_xlabel('Scaling Factors', fontsize=11)
    ax.set_ylabel('Total cost', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cost_breakdown_memory_error_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: cost_breakdown_memory_error_{timestamp}.png")
    plt.close()

    # Figure 4: Memory Utilization Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Average memory utilization vs error scaling
    for C_gpu_scale in sorted(df['C_gpu_scale'].unique()):
        subset = df[df['C_gpu_scale'] == C_gpu_scale]
        axes[0].plot(subset['psi_error'], subset['avg_memory_util'],
                    marker='o', label=f'C_gpu={C_gpu_scale:.1f}×', linewidth=2)

    axes[0].set_xlabel('Error Scaling (psi_error)', fontsize=11)
    axes[0].set_ylabel('Average Memory Utilization', fontsize=11)
    axes[0].set_title('Memory Utilization vs Error Scaling\n(Different Memory Capacities)', fontweight='bold')
    axes[0].legend(title='Memory Scaling', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1.0, color='r', linestyle='--', label='Full Capacity', alpha=0.5)

    # Total GPUs vs memory scaling
    for psi_e in sorted(df['psi_error'].unique())[::2]:
        subset = df[df['psi_error'] == psi_e]
        axes[1].plot(subset['C_gpu_scale'], subset['total_gpus'],
                    marker='s', label=f'ψ_ε={psi_e:.1f}', linewidth=2)

    axes[1].set_xlabel('Memory Capacity Scaling (C_gpu)', fontsize=11)
    axes[1].set_ylabel('Total Number of GPUs', fontsize=11)
    axes[1].set_title('GPU Count vs Memory Capacity\n(Different Error Scalings)', fontweight='bold')
    axes[1].legend(title='Error Scaling', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'memory_analysis_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: memory_analysis_{timestamp}.png")
    plt.close()
  # Figure 7: Tensor Parallelism Distribution Stacked Bar Chart
    fig, ax = plt.subplots(figsize=(16, 8))

    # Select representative scenarios across the parameter space
    # Sort by both parameters and sample evenly
    df_sorted = df.sort_values(['C_gpu_scale', 'psi_error'])

    # Sample every nth scenario to get representative points (aim for ~15-20 scenarios)
    num_scenarios = min(20, len(df_sorted))
    step = max(1, len(df_sorted) // num_scenarios)
    selected_scenarios = df_sorted.iloc[::step]

    # Create labels showing both memory capacity and error threshold
    x_labels = [f"C_gpu={row['C_gpu_scale']:.1f}×\nψ_e={row['psi_error']:.1f}"
                for _, row in selected_scenarios.iterrows()]

    x_pos = np.arange(len(selected_scenarios))
    width = 0.7

    # Create stacked bars for each TP degree
    ax.bar(x_pos, selected_scenarios['tp_1_count'], width,
           label='TP=1 (No Parallelism)', color='#3498db')

    ax.bar(x_pos, selected_scenarios['tp_2_count'], width,
           bottom=selected_scenarios['tp_1_count'],
           label='TP=2', color='#2ecc71')

    ax.bar(x_pos, selected_scenarios['tp_4_count'], width,
           bottom=selected_scenarios['tp_1_count'] + selected_scenarios['tp_2_count'],
           label='TP=4', color='#f39c12')

    ax.bar(x_pos, selected_scenarios['tp_8_count'], width,
           bottom=selected_scenarios['tp_1_count'] + selected_scenarios['tp_2_count'] + selected_scenarios['tp_4_count'],
           label='TP=8 (Max Parallelism)', color='#e74c3c')

    # Add average TP degree as line plot on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(x_pos, selected_scenarios['avg_tp_degree'],
             color='black', marker='D', linewidth=2, markersize=6,
             label='Avg TP Degree', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Average TP Degree', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=10)
    ax2.set_ylim(0, max(selected_scenarios['avg_tp_degree']) * 1.2)

    # Formatting
    ax.set_xlabel('Memory Capacity Scaling (C_gpu) & Error Threshold Scaling (ψ_e)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Configurations', fontsize=12, fontweight='bold')
    ax.set_title('Tensor Parallelism Distribution Across Memory Capacity and Error Threshold Scenarios',
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc='upper left', fontsize=10, framealpha=0.9)

    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'tp_distribution_stacked_bar_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: tp_distribution_stacked_bar_{timestamp}.png")
    plt.close()

    # Figure 8: Tensor Parallelism Heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Average TP degree heatmap
    tp_pivot = df.pivot_table(values='avg_tp_degree', index='C_gpu_scale', columns='psi_error')
    sns.heatmap(tp_pivot, annot=True, fmt='.2f', cmap='viridis', ax=axes[0, 0],
                cbar_kws={'label': 'Average TP Degree'})
    axes[0, 0].set_title('Average Tensor Parallelism Degree', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Error Threshold Scaling (ψ_error)')
    axes[0, 0].set_ylabel('Memory Capacity Scaling (C_gpu)')

    # TP=1 count heatmap
    tp1_pivot = df.pivot_table(values='tp_1_count', index='C_gpu_scale', columns='psi_error')
    sns.heatmap(tp1_pivot, annot=True, fmt='.0f', cmap='Blues', ax=axes[0, 1],
                cbar_kws={'label': 'Count'})
    axes[0, 1].set_title('TP=1 Configurations (No Parallelism)', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Error Threshold Scaling (ψ_error)')
    axes[0, 1].set_ylabel('Memory Capacity Scaling (C_gpu)')

    # TP=2 count heatmap
    tp2_pivot = df.pivot_table(values='tp_2_count', index='C_gpu_scale', columns='psi_error')
    sns.heatmap(tp2_pivot, annot=True, fmt='.0f', cmap='Greens', ax=axes[0, 2],
                cbar_kws={'label': 'Count'})
    axes[0, 2].set_title('TP=2 Configurations', fontweight='bold', fontsize=12)
    axes[0, 2].set_xlabel('Error Threshold Scaling (ψ_error)')
    axes[0, 2].set_ylabel('Memory Capacity Scaling (C_gpu)')

    # TP=4 count heatmap
    tp4_pivot = df.pivot_table(values='tp_4_count', index='C_gpu_scale', columns='psi_error')
    sns.heatmap(tp4_pivot, annot=True, fmt='.0f', cmap='Oranges', ax=axes[1, 0],
                cbar_kws={'label': 'Count'})
    axes[1, 0].set_title('TP=4 Configurations', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Error Threshold Scaling (ψ_error)')
    axes[1, 0].set_ylabel('Memory Capacity Scaling (C_gpu)')

    # TP=8 count heatmap
    tp8_pivot = df.pivot_table(values='tp_8_count', index='C_gpu_scale', columns='psi_error')
    sns.heatmap(tp8_pivot, annot=True, fmt='.0f', cmap='Reds', ax=axes[1, 1],
                cbar_kws={'label': 'Count'})
    axes[1, 1].set_title('TP=8 Configurations (Max Parallelism)', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Error Threshold Scaling (ψ_error)')
    axes[1, 1].set_ylabel('Memory Capacity Scaling (C_gpu)')

    # Total TP configurations heatmap
    tp_total_pivot = df.pivot_table(values='num_tp_configs', index='C_gpu_scale', columns='psi_error')
    sns.heatmap(tp_total_pivot, annot=True, fmt='.0f', cmap='Purples', ax=axes[1, 2],
                cbar_kws={'label': 'Total Configs'})
    axes[1, 2].set_title('Total TP Configurations', fontweight='bold', fontsize=12)
    axes[1, 2].set_xlabel('Error Threshold Scaling (ψ_error)')
    axes[1, 2].set_ylabel('Memory Capacity Scaling (C_gpu)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'tp_selection_heatmaps_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: tp_selection_heatmaps_{timestamp}.png")
    plt.close()

    print("Visualization complete!")

if __name__ == "__main__":
    results_df = main()