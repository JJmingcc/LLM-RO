"""
Sensitivity Analysis for GPU Rental Cost and Budget
Analyzes the impact of varying GPU rental cost (p_c) and budget threshold (delta)
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

def run_sensitivity_analysis(p_c_scale, delta_scale, base_data, time_limit=300, mip_gap=0.01):
    """
    Run optimization with scaled GPU rental cost and budget threshold.

    Args:
        p_c_scale: Scaling factor for GPU rental costs (p_c)
        delta_scale: Scaling factor for budget threshold (delta)
        base_data: Base data object with original parameters
        time_limit: Optimization time limit in seconds
        mip_gap: MIP gap tolerance

    Returns:
        Dictionary with results including costs, allocations, and metrics
    """
    # Create a copy of data with scaled parameters
    data = copy.deepcopy(base_data)

    # Scale GPU rental costs and budget threshold
    data.p_c = base_data.p_c * p_c_scale
    data.delta = base_data.delta * delta_scale

    # Build and solve optimization
    start_time = time.time()
    try:
        optimizer = LLMInferenceOptimizer(data)
        solution = optimizer.build_and_solve_optimization_problem(time_limit=time_limit, mip_gap=mip_gap)
        solve_time = time.time() - start_time

        if solution is None or solution['status'] not in ['OPTIMAL', 'TIME_LIMIT']:
            return {
                'p_c_scale': p_c_scale,
                'delta_scale': delta_scale,
                'status': 'INFEASIBLE' if solution is None else solution.get('status', 'INFEASIBLE'),
                'solve_time': solve_time,
                'total_cost': None,
                'gap': None
            }

        # Extract detailed metrics
        results = extract_metrics(optimizer, solution, data, p_c_scale, delta_scale, solve_time)
        return results

    except Exception as e:
        print(f"Error for p_c_scale={p_c_scale}, delta_scale={delta_scale}: {str(e)}")
        return {
            'p_c_scale': p_c_scale,
            'delta_scale': delta_scale,
            'status': 'ERROR',
            'error_message': str(e),
            'solve_time': time.time() - start_time,
            'total_cost': None,
            'gap': None
        }

def extract_metrics(optimizer, solution, data, p_c_scale, delta_scale, solve_time):
    """
    Extract detailed metrics from optimization solution.

    Args:
        optimizer: LLMInferenceOptimizer instance
        solution: Solution dictionary
        data: Data object used for optimization
        p_c_scale: GPU rental cost scaling factor
        delta_scale: Budget scaling factor
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

    # Calculate budget utilization
    budget_utilization = solution['objective'] / data.delta if data.delta > 0 else 0

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
        'p_c_scale': p_c_scale,
        'delta_scale': delta_scale,
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
        'C1_pct': C1 / solution['objective'] * 100 if solution['objective'] > 0 else 0,
        'C2_pct': C2 / solution['objective'] * 100 if solution['objective'] > 0 else 0,
        'C3_pct': C3 / solution['objective'] * 100 if solution['objective'] > 0 else 0,
        'C4_pct': C4 / solution['objective'] * 100 if solution['objective'] > 0 else 0,

        # Resource metrics
        'num_allocations': len(solution.get('x', {})),
        'num_gpu_configs': len(solution.get('y', {})),
        'total_gpus': total_gpus,
        'num_unmet_queries': len(solution.get('u', {})),
        'total_unmet_fraction': total_unmet,

        # Performance metrics
        'avg_delay': avg_delay,
        'avg_error': avg_error,

        # Budget metrics
        'budget_threshold': data.delta,
        'budget_utilization': budget_utilization,
        'budget_slack': data.delta - solution['objective'],

        # Scaled parameters
        'scaled_p_c_mean': np.mean(data.p_c),
        'scaled_delta': data.delta,

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
    print("SENSITIVITY ANALYSIS: GPU RENTAL COST & BUDGET")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Generate base data
    print("Generating base data using DataGenerator...")
    generator = DataGenerator(seed=42)
    base_data = generator.generate()
    generator.print_summary(base_data)

    # Define scaling factors
    p_c_scale_values = [0.4,0.45,0.5,0.55,0.6,0.65]
    delta_scale_values = [0.3,0.35,0.4,0.45,0.5]

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS PARAMETERS")
    print("="*80)
    print(f"Base GPU rental costs (p_c): {base_data.p_c}")
    print(f"Base budget threshold (delta): {base_data.delta}")
    print(f"\nGPU rental cost scaling factors (p_c_scale): {p_c_scale_values}")
    print(f"Budget scaling factors (delta_scale): {delta_scale_values}")
    print(f"Total scenarios: {len(p_c_scale_values)} × {len(delta_scale_values)} = {len(p_c_scale_values) * len(delta_scale_values)}")

    # Run sensitivity analysis
    results_list = []
    counter = 0
    total = len(p_c_scale_values) * len(delta_scale_values)

    print("\n" + "="*80)
    print("Running sensitivity analysis...")
    print("="*80)

    for p_c_scale in p_c_scale_values:
        for delta_scale in delta_scale_values:
            counter += 1
            print(f"\n[{counter}/{total}] p_c_scale={p_c_scale:.1f}, delta_scale={delta_scale:.1f}")

            result = run_sensitivity_analysis(p_c_scale, delta_scale, base_data, time_limit=300, mip_gap=0.01)
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
        csv_filename = os.path.join(output_dir, f'sensitivity_cost_budget_{timestamp}.csv')
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
    cost_pivot = df.pivot_table(values='total_cost', index='p_c_scale', columns='delta_scale')
    sns.heatmap(cost_pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0, 0],
                cbar_kws={'label': 'Total Cost ($)'})
    axes[0, 0].set_title('Total Cost Heatmap', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Budget Scaling (delta_scale)')
    axes[0, 0].set_ylabel('GPU Rental Cost Scaling (p_c)')

    # Rental Cost Heatmap
    rental_pivot = df.pivot_table(values='C1_rental', index='p_c_scale', columns='delta_scale')
    sns.heatmap(rental_pivot, annot=True, fmt='.0f', cmap='Blues', ax=axes[0, 1],
                cbar_kws={'label': 'Rental Cost ($)'})
    axes[0, 1].set_title('GPU Rental Cost (C1)', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Budget Scaling (delta_scale)')
    axes[0, 1].set_ylabel('GPU Rental Cost Scaling (p_c)')

    # Unmet Demand Cost Heatmap
    unmet_pivot = df.pivot_table(values='C4_unmet', index='p_c_scale', columns='delta_scale')
    sns.heatmap(unmet_pivot, annot=True, fmt='.0f', cmap='Reds', ax=axes[1, 0],
                cbar_kws={'label': 'Unmet Demand Cost ($)'})
    axes[1, 0].set_title('Unmet Demand Penalty (C4)', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Budget Scaling (delta_scale)')
    axes[1, 0].set_ylabel('GPU Rental Cost Scaling (p_c)')

    # Budget Utilization Heatmap
    budget_pivot = df.pivot_table(values='budget_utilization', index='p_c_scale', columns='delta_scale')
    sns.heatmap(budget_pivot, annot=True, fmt='.2f', cmap='Greens', ax=axes[1, 1],
                cbar_kws={'label': 'Budget Utilization Ratio'})
    axes[1, 1].set_title('Budget Utilization (Cost/Budget)', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Budget Scaling (delta_scale)')
    axes[1, 1].set_ylabel('GPU Rental Cost Scaling (p_c)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmaps_cost_budget_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: heatmaps_cost_budget_{timestamp}.png")
    plt.close()

    # Figure 2: Cost Component Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Line plot for fixed rental cost scaling
    for p_c_scale in sorted(df['p_c_scale'].unique()):
        subset = df[df['p_c_scale'] == p_c_scale]
        axes[0].plot(subset['delta_scale'], subset['total_cost'],
                    marker='o', label=f'p_c={p_c_scale:.1f}×', linewidth=2)

    axes[0].set_xlabel('Budget Scaling (delta_scale)', fontsize=11)
    axes[0].set_ylabel('Total Cost ($)', fontsize=11)
    axes[0].set_title('Total Cost vs Budget Scaling\n(Different GPU Rental Costs)', fontweight='bold')
    axes[0].legend(title='Rental Cost Scaling', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Line plot for fixed budget scaling
    for delta_scale in sorted(df['delta_scale'].unique())[::2]:  # Show every other budget value
        subset = df[df['delta_scale'] == delta_scale]
        axes[1].plot(subset['p_c_scale'], subset['total_cost'],
                    marker='s', label=f'δ={delta_scale:.1f}×', linewidth=2)

    axes[1].set_xlabel('GPU Rental Cost Scaling (p_c)', fontsize=11)
    axes[1].set_ylabel('Total Cost ($)', fontsize=11)
    axes[1].set_title('Total Cost vs Rental Cost Scaling\n(Different Budget Levels)', fontweight='bold')
    axes[1].legend(title='Budget Scaling', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cost_trends_cost_budget_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: cost_trends_cost_budget_{timestamp}.png")
    plt.close()

    # Figure 3: Cost Component Stacked Bar Chart
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Select representative scenarios
    selected_scenarios = df.sort_values('total_cost').iloc[::len(df)//10][:10]  # 10 representative points

    x_labels = [f"p_c={row['p_c_scale']:.1f}×\nδ={row['delta_scale']:.1f}×"
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

    ax.set_xlabel('Scaling Factors', fontsize=11)
    ax.set_ylabel('Cost ($)', fontsize=11)
    ax.set_title('Cost Component Breakdown (Selected Scenarios)', fontweight='bold', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cost_breakdown_cost_budget_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: cost_breakdown_cost_budget_{timestamp}.png")
    plt.close()

    # Figure 4: Budget Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Budget utilization vs rental cost scaling
    for delta_scale in sorted(df['delta_scale'].unique()):
        subset = df[df['delta_scale'] == delta_scale]
        axes[0].plot(subset['p_c_scale'], subset['budget_utilization'],
                    marker='o', label=f'δ={delta_scale:.1f}×', linewidth=2)

    axes[0].set_xlabel('GPU Rental Cost Scaling (p_c)', fontsize=11)
    axes[0].set_ylabel('Budget Utilization Ratio', fontsize=11)
    axes[0].set_title('Budget Utilization vs Rental Cost\n(Different Budget Levels)', fontweight='bold')
    axes[0].legend(title='Budget Scaling', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1.0, color='r', linestyle='--', label='Full Budget', alpha=0.5)

    # Total GPUs vs rental cost scaling
    for delta_scale in sorted(df['delta_scale'].unique())[::2]:
        subset = df[df['delta_scale'] == delta_scale]
        axes[1].plot(subset['p_c_scale'], subset['total_gpus'],
                    marker='s', label=f'δ={delta_scale:.1f}×', linewidth=2)

    axes[1].set_xlabel('GPU Rental Cost Scaling (p_c)', fontsize=11)
    axes[1].set_ylabel('Total Number of GPUs', fontsize=11)
    axes[1].set_title('GPU Count vs Rental Cost\n(Different Budget Levels)', fontweight='bold')
    axes[1].legend(title='Budget Scaling', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'budget_analysis_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: budget_analysis_{timestamp}.png")
    plt.close()

    # Figure 5: Rental Cost Percentage Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # C1 percentage vs rental cost scaling
    for delta_scale in sorted(df['delta_scale'].unique())[::2]:
        subset = df[df['delta_scale'] == delta_scale]
        axes[0].plot(subset['p_c_scale'], subset['C1_pct'],
                    marker='o', label=f'δ={delta_scale:.1f}×', linewidth=2)

    axes[0].set_xlabel('GPU Rental Cost Scaling (p_c)', fontsize=11)
    axes[0].set_ylabel('Rental Cost Percentage (%)', fontsize=11)
    axes[0].set_title('C1 Percentage vs Rental Cost\n(Different Budget Levels)', fontweight='bold')
    axes[0].legend(title='Budget Scaling', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Unmet demand fraction vs rental cost scaling
    for delta_scale in sorted(df['delta_scale'].unique())[::2]:
        subset = df[df['delta_scale'] == delta_scale]
        axes[1].plot(subset['p_c_scale'], subset['total_unmet_fraction'],
                    marker='s', label=f'δ={delta_scale:.1f}×', linewidth=2)

    axes[1].set_xlabel('GPU Rental Cost Scaling (p_c)', fontsize=11)
    axes[1].set_ylabel('Total Unmet Demand Fraction', fontsize=11)
    axes[1].set_title('Unmet Demand vs Rental Cost\n(Different Budget Levels)', fontweight='bold')
    axes[1].legend(title='Budget Scaling', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'rental_cost_analysis_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: rental_cost_analysis_{timestamp}.png")
    plt.close()

    print("Visualization complete!")

if __name__ == "__main__":
    results_df = main()