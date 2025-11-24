import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from RODIU_LLM import DataGenerator

# Generate data
generator = DataGenerator(seed=42)
data = generator.generate()

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Define query type for visualization (we'll show one query type as example)
query_idx = 0  # Summarization

# ==================== PLOT 1: Delay Heatmap (for query type 0) ====================
ax1 = axes[0, 0]
delay_matrix = data.d[query_idx, :, :]  # Shape: (J, K)
sns.heatmap(delay_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=data.gpu_tiers, yticklabels=data.model_names,
            ax=ax1, cbar_kws={'label': 'Delay (ms/token)'})
ax1.set_title(f'Processing Delay for {data.query_types[query_idx]}', fontsize=14, fontweight='bold')
ax1.set_xlabel('GPU Configuration', fontsize=12)
ax1.set_ylabel('Model', fontsize=12)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ==================== PLOT 2: Error Rate Heatmap (for query type 0) ====================
ax2 = axes[0, 1]
error_matrix = data.e[query_idx, :, :]  # Shape: (J, K)
sns.heatmap(error_matrix, annot=True, fmt='.4f', cmap='RdYlGn_r',
            xticklabels=data.gpu_tiers, yticklabels=data.model_names,
            ax=ax2, cbar_kws={'label': 'Error Rate'})
ax2.set_title(f'Error Rate for {data.query_types[query_idx]}', fontsize=14, fontweight='bold')
ax2.set_xlabel('GPU Configuration', fontsize=12)
ax2.set_ylabel('Model', fontsize=12)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ==================== PLOT 3: Delay by Model across GPU configs ====================
ax3 = axes[1, 0]
for j in range(data.J):
    ax3.plot(range(data.K), data.d[query_idx, j, :], marker='o', label=data.model_names[j], linewidth=2)
ax3.set_xlabel('GPU Configuration', fontsize=12)
ax3.set_ylabel('Delay (ms/token)', fontsize=12)
ax3.set_title(f'Delay Comparison Across GPU Configs for {data.query_types[query_idx]}', fontsize=14, fontweight='bold')
ax3.set_xticks(range(data.K))
ax3.set_xticklabels(data.gpu_tiers, rotation=45, ha='right')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# ==================== PLOT 4: Error Rate by Model across GPU configs ====================
ax4 = axes[1, 1]
for j in range(data.J):
    ax4.plot(range(data.K), data.e[query_idx, j, :], marker='s', label=data.model_names[j], linewidth=2)
ax4.set_xlabel('GPU Configuration', fontsize=12)
ax4.set_ylabel('Error Rate', fontsize=12)
ax4.set_title(f'Error Rate Comparison Across GPU Configs for {data.query_types[query_idx]}', fontsize=14, fontweight='bold')
ax4.set_xticks(range(data.K))
ax4.set_xticklabels(data.gpu_tiers, rotation=45, ha='right')
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('delay_error_analysis.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'delay_error_analysis.png'")

# ==================== ADDITIONAL PLOTS: All Query Types ====================
fig2, axes2 = plt.subplots(3, 2, figsize=(20, 18))

for i in range(data.I):
    row = i // 2
    col = i % 2
    ax = axes2[row, col]

    # Plot delay for each model
    for j in range(data.J):
        ax.plot(range(data.K), data.d[i, j, :], marker='o', label=data.model_names[j], linewidth=2, alpha=0.7)

    ax.set_xlabel('GPU Configuration', fontsize=11)
    ax.set_ylabel('Delay (ms/token)', fontsize=11)
    ax.set_title(f'Delay: {data.query_types[i]}', fontsize=13, fontweight='bold')
    ax.set_xticks(range(data.K))
    ax.set_xticklabels(data.gpu_tiers, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('delay_all_queries.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'delay_all_queries.png'")

# ==================== ERROR RATE: All Query Types ====================
fig3, axes3 = plt.subplots(3, 2, figsize=(20, 18))

for i in range(data.I):
    row = i // 2
    col = i % 2
    ax = axes3[row, col]

    # Plot error rate for each model
    for j in range(data.J):
        ax.plot(range(data.K), data.e[i, j, :], marker='s', label=data.model_names[j], linewidth=2, alpha=0.7)

    ax.set_xlabel('GPU Configuration', fontsize=11)
    ax.set_ylabel('Error Rate', fontsize=11)
    ax.set_title(f'Error Rate: {data.query_types[i]}', fontsize=13, fontweight='bold')
    ax.set_xticks(range(data.K))
    ax.set_xticklabels(data.gpu_tiers, rotation=45, ha='right', fontsize=9)

    # Add epsilon threshold line
    ax.axhline(y=data.epsilon[i], color='red', linestyle='--', linewidth=2, label=f'Threshold (ε={data.epsilon[i]:.3f})')

    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_rate_all_queries.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'error_rate_all_queries.png'")

# ==================== SUMMARY STATISTICS ====================
print("\n" + "="*80)
print("DELAY AND ERROR RATE STATISTICS")
print("="*80)

print("\n1. DELAY STATISTICS (ms/token):")
print(f"   {'Model':20s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'Std':>10s}")
print("   " + "-"*60)
for j in range(data.J):
    delays = data.d[:, j, :].flatten()
    print(f"   {data.model_names[j]:20s} {delays.min():10.2f} {delays.max():10.2f} {delays.mean():10.2f} {delays.std():10.2f}")

print("\n2. ERROR RATE STATISTICS:")
print(f"   {'Model':20s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'Std':>10s}")
print("   " + "-"*60)
for j in range(data.J):
    errors = data.e[:, j, :].flatten()
    print(f"   {data.model_names[j]:20s} {errors.min():10.4f} {errors.max():10.4f} {errors.mean():10.4f} {errors.std():10.4f}")

print("\n3. GPU PRECISION IMPACT:")
print(f"   {'GPU Config':20s} {'Precision':>12s} {'Avg Delay':>12s} {'Avg Error':>12s}")
print("   " + "-"*60)
for k in range(data.K):
    precision = 'FP16' if 'FP16' in data.gpu_tiers[k] else ('INT8' if 'INT8' in data.gpu_tiers[k] else 'INT4')
    avg_delay = data.d[:, :, k].mean()
    avg_error = data.e[:, :, k].mean()
    print(f"   {data.gpu_tiers[k]:20s} {precision:>12s} {avg_delay:12.2f} {avg_error:12.4f}")

print("\n4. RELATIONSHIP ANALYSIS:")
print("\n   A. Delay vs GPU Compute Power:")
print(f"      {'GPU':20s} {'Compute (TFLOPS)':>18s} {'Avg Delay':>12s} {'Speedup':>10s}")
print("      " + "-"*65)
baseline_delay = data.d[:, :, 0].mean()
for k in range(data.K):
    avg_delay = data.d[:, :, k].mean()
    speedup = baseline_delay / avg_delay
    print(f"      {data.gpu_tiers[k]:20s} {data.P_gpu[k]:18.1f} {avg_delay:12.2f} {speedup:10.2f}x")

print("\n   B. Error Rate vs Model Size:")
print(f"      {'Model':20s} {'Size (GB)':>12s} {'Avg Error':>12s}")
print("      " + "-"*50)
for j in range(data.J):
    avg_error = data.e[:, j, :].mean()
    print(f"      {data.model_names[j]:20s} {data.B[j]:12.1f} {avg_error:12.4f}")

print("\n" + "="*80)
print("Visualization complete! Check the generated PNG files.")
print("="*80)