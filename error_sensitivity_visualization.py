# ============================================================================
# VISUALIZATION: Error Impact Spacing vs Error Rate Threshold Analysis
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def visualize_error_sensitivity_results(df_results: pd.DataFrame, save_figures: bool = True):
    """
    Create comprehensive visualizations for the error sensitivity analysis.

    Args:
        df_results: DataFrame containing the analysis results
        save_figures: Whether to save figures to files
    """
    # Filter successful results
    df_success = df_results[df_results['status'].isin(['OPTIMAL', 'TIME_LIMIT'])].copy()

    print(f"Visualizing {len(df_success)} successful scenarios out of {len(df_results)} total")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # ========================================================================
    # 1. Total Cost Heatmap
    # ========================================================================
    ax1 = plt.subplot(2, 3, 1)
    cost_pivot = df_success.pivot_table(
        values='total_cost',
        index='epsilon_scale',
        columns='delta_error',
        aggfunc='mean'
    )

    sns.heatmap(cost_pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                cbar_kws={'label': 'Total Cost ($)'}, ax=ax1)
    ax1.set_title('Total Cost vs Error Parameters\n(Fixed Delay Impact = 0.2)',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Error Impact Spacing (Δ_error)', fontsize=10)
    ax1.set_ylabel('Error Threshold Scale (ε_scale)', fontsize=10)

    # ========================================================================
    # 2. Cost Component Breakdown
    # ========================================================================
    ax2 = plt.subplot(2, 3, 2)

    # Calculate average costs for each component
    cost_components = ['inference_cost', 'storage_cost', 'rental_cost', 'penalty_cost']
    avg_costs = df_success[cost_components].mean()

    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax2.bar(range(len(cost_components)), avg_costs, color=colors, alpha=0.7)

    ax2.set_xticks(range(len(cost_components)))
    ax2.set_xticklabels(['Inference', 'Storage', 'Rental', 'Penalty'], rotation=45, ha='right')
    ax2.set_ylabel('Average Cost ($)', fontsize=10)
    ax2.set_title('Average Cost Component Breakdown', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=9)

    # ========================================================================
    # 3. Total Cost by Error Impact Spacing
    # ========================================================================
    ax3 = plt.subplot(2, 3, 3)

    for epsilon_scale in sorted(df_success['epsilon_scale'].unique()):
        subset = df_success[df_success['epsilon_scale'] == epsilon_scale]
        ax3.plot(subset['delta_error'], subset['total_cost'],
                marker='o', label=f'ε_scale={epsilon_scale:.2f}', linewidth=2)

    ax3.set_xlabel('Error Impact Spacing (Δ_error)', fontsize=10)
    ax3.set_ylabel('Total Cost ($)', fontsize=10)
    ax3.set_title('Total Cost vs Error Impact Spacing\n(Different Error Threshold Scales)',
                  fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ========================================================================
    # 4. Total Cost by Error Threshold Scale
    # ========================================================================
    ax4 = plt.subplot(2, 3, 4)

    for delta_error in sorted(df_success['delta_error'].unique()):
        subset = df_success[df_success['delta_error'] == delta_error]
        ax4.plot(subset['epsilon_scale'], subset['total_cost'],
                marker='s', label=f'Δ_error={delta_error:.2f}', linewidth=2)

    ax4.set_xlabel('Error Threshold Scale (ε_scale)', fontsize=10)
    ax4.set_ylabel('Total Cost ($)', fontsize=10)
    ax4.set_title('Total Cost vs Error Threshold Scale\n(Different Error Impact Spacings)',
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ========================================================================
    # 5. Solve Time Analysis
    # ========================================================================
    ax5 = plt.subplot(2, 3, 5)

    solve_time_pivot = df_success.pivot_table(
        values='solve_time',
        index='epsilon_scale',
        columns='delta_error',
        aggfunc='mean'
    )

    sns.heatmap(solve_time_pivot, annot=True, fmt='.1f', cmap='Blues',
                cbar_kws={'label': 'Solve Time (s)'}, ax=ax5)
    ax5.set_title('Average Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Error Impact Spacing (Δ_error)', fontsize=10)
    ax5.set_ylabel('Error Threshold Scale (ε_scale)', fontsize=10)

    # ========================================================================
    # 6. Cost Sensitivity Metrics
    # ========================================================================
    ax6 = plt.subplot(2, 3, 6)

    # Calculate cost sensitivity to each parameter
    sensitivity_data = []

    for delta_error in sorted(df_success['delta_error'].unique()):
        subset = df_success[df_success['delta_error'] == delta_error]
        cost_range = subset['total_cost'].max() - subset['total_cost'].min()
        sensitivity_data.append({
            'Parameter': f'Δ_err={delta_error:.2f}',
            'Cost Range': cost_range,
            'Type': 'Error Impact'
        })

    for epsilon_scale in sorted(df_success['epsilon_scale'].unique()):
        subset = df_success[df_success['epsilon_scale'] == epsilon_scale]
        cost_range = subset['total_cost'].max() - subset['total_cost'].min()
        sensitivity_data.append({
            'Parameter': f'ε_scl={epsilon_scale:.2f}',
            'Cost Range': cost_range,
            'Type': 'Threshold Scale'
        })

    df_sensitivity = pd.DataFrame(sensitivity_data)

    # Create grouped bar chart
    param_labels = df_sensitivity['Parameter'].tolist()
    colors_list = ['#e74c3c' if t == 'Error Impact' else '#3498db'
                   for t in df_sensitivity['Type']]

    bars = ax6.barh(range(len(param_labels)), df_sensitivity['Cost Range'],
                    color=colors_list, alpha=0.7)

    ax6.set_yticks(range(len(param_labels)))
    ax6.set_yticklabels(param_labels, fontsize=8)
    ax6.set_xlabel('Total Cost Range ($)', fontsize=10)
    ax6.set_title('Cost Sensitivity to Parameters\n(Range across other parameter)',
                  fontsize=12, fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.7, label='Error Impact Spacing'),
        Patch(facecolor='#3498db', alpha=0.7, label='Error Threshold Scale')
    ]
    ax6.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()

    if save_figures:
        plt.savefig('error_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        print("Figure saved as 'error_sensitivity_analysis.png'")

    plt.show()

    # ========================================================================
    # Additional Statistics
    # ========================================================================
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS STATISTICS")
    print("="*80)

    print(f"\nTotal Cost Range: ${df_success['total_cost'].min():,.2f} - ${df_success['total_cost'].max():,.2f}")
    print(f"Total Cost Mean: ${df_success['total_cost'].mean():,.2f}")
    print(f"Total Cost Std: ${df_success['total_cost'].std():,.2f}")

    print(f"\nBest scenario (lowest cost):")
    best_idx = df_success['total_cost'].idxmin()
    best_row = df_success.loc[best_idx]
    print(f"  Delta_error: {best_row['delta_error']:.2f}")
    print(f"  Epsilon_scale: {best_row['epsilon_scale']:.2f}")
    print(f"  Total cost: ${best_row['total_cost']:,.2f}")

    print(f"\nWorst scenario (highest cost):")
    worst_idx = df_success['total_cost'].idxmax()
    worst_row = df_success.loc[worst_idx]
    print(f"  Delta_error: {worst_row['delta_error']:.2f}")
    print(f"  Epsilon_scale: {worst_row['epsilon_scale']:.2f}")
    print(f"  Total cost: ${worst_row['total_cost']:,.2f}")

    # Calculate interaction effects
    print(f"\nParameter Effects:")
    print(f"  Effect of Error Impact Spacing (Δ_error):")
    for delta_error in sorted(df_success['delta_error'].unique()):
        avg_cost = df_success[df_success['delta_error'] == delta_error]['total_cost'].mean()
        print(f"    Δ_error={delta_error:.2f}: Avg cost = ${avg_cost:,.2f}")

    print(f"\n  Effect of Error Threshold Scale (ε_scale):")
    for epsilon_scale in sorted(df_success['epsilon_scale'].unique()):
        avg_cost = df_success[df_success['epsilon_scale'] == epsilon_scale]['total_cost'].mean()
        print(f"    ε_scale={epsilon_scale:.2f}: Avg cost = ${avg_cost:,.2f}")

def create_3d_surface_plot(df_results: pd.DataFrame, save_figure: bool = True):
    """
    Create a 3D surface plot for total cost.

    Args:
        df_results: DataFrame containing the analysis results
        save_figure: Whether to save the figure
    """
    from mpl_toolkits.mplot3d import Axes3D

    df_success = df_results[df_results['status'].isin(['OPTIMAL', 'TIME_LIMIT'])].copy()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data
    cost_pivot = df_success.pivot_table(
        values='total_cost',
        index='epsilon_scale',
        columns='delta_error',
        aggfunc='mean'
    )

    X = cost_pivot.columns.values
    Y = cost_pivot.index.values
    X, Y = np.meshgrid(X, Y)
    Z = cost_pivot.values

    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

    # Add wireframe for better depth perception
    ax.plot_wireframe(X, Y, Z, color='black', alpha=0.2, linewidth=0.5)

    ax.set_xlabel('Error Impact Spacing (Δ_error)', fontsize=10, labelpad=10)
    ax.set_ylabel('Error Threshold Scale (ε_scale)', fontsize=10, labelpad=10)
    ax.set_zlabel('Total Cost ($)', fontsize=10, labelpad=10)
    ax.set_title('3D Surface: Total Cost vs Error Parameters\n(Fixed Delay Impact = 0.2)',
                 fontsize=12, fontweight='bold', pad=20)

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Total Cost ($)')

    # Adjust viewing angle
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()

    if save_figure:
        plt.savefig('error_sensitivity_3d.png', dpi=300, bbox_inches='tight')
        print("3D figure saved as 'error_sensitivity_3d.png'")

    plt.show()

if __name__ == "__main__":
    # Load results
    df_results = pd.read_csv('error_sensitivity_results.csv')

    # Create visualizations
    visualize_error_sensitivity_results(df_results, save_figures=True)
    create_3d_surface_plot(df_results, save_figure=True)