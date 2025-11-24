import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from RDDU_LLM_inference_opt import DataGenerator, LLMInferenceOptimizer

# Generate data and solve optimization
print("Generating problem data...")
generator = DataGenerator(seed=42)
data = generator.generate()

print("\nSolving optimization model...")
optimizer = LLMInferenceOptimizer(data)
solution = optimizer.build_and_solve_optimization_problem(time_limit=300, mip_gap=0.01)

# Extract w values (TP degree selection)
print("\nExtracting TP degree choices (w variable)...")
w_vals = {}
for j in range(data.J):
    for k in range(data.K):
        for n in data.TP_degrees:
            w_val = optimizer.vars['w'][j, k, n].X
            if w_val > 0.5:  # Binary variable, should be 0 or 1
                w_vals[(j, k)] = n

# Also get q values to see which configurations are active
q_vals = {}
for j in range(data.J):
    for k in range(data.K):
        q_val = optimizer.vars['q'][j, k].X
        if q_val > 0.5:
            q_vals[(j, k)] = 1

# Also get y values to see the number of GPUs
y_vals = {}
for j in range(data.J):
    for k in range(data.K):
        y_val = optimizer.vars['y'][j, k].X
        if y_val > 0.01:
            y_vals[(j, k)] = int(y_val)

print(f"\nActive configurations (q=1): {len(q_vals)}")
print(f"TP degree selections (w=1): {len(w_vals)}")
print(f"GPU allocations (y>0): {len(y_vals)}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Heatmap of TP degree selection
tp_matrix = np.zeros((data.J, data.K))
for (j, k), n in w_vals.items():
    tp_matrix[j, k] = n

ax1 = axes[0, 0]
sns.heatmap(tp_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
            xticklabels=data.gpu_tiers, yticklabels=data.model_names,
            cbar_kws={'label': 'TP Degree (n)'}, ax=ax1, vmin=0, vmax=8)
ax1.set_title('TP Degree Selection per Model-GPU Configuration', fontsize=12, fontweight='bold')
ax1.set_xlabel('GPU Tier', fontsize=10)
ax1.set_ylabel('Model', fontsize=10)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)

# 2. Bar chart showing distribution of TP degrees chosen
tp_degree_counts = {}
for n in data.TP_degrees:
    tp_degree_counts[n] = sum(1 for val in w_vals.values() if val == n)

ax2 = axes[0, 1]
bars = ax2.bar([str(n) for n in data.TP_degrees],
               [tp_degree_counts.get(n, 0) for n in data.TP_degrees],
               color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
ax2.set_title('Distribution of TP Degree Choices', fontsize=12, fontweight='bold')
ax2.set_xlabel('TP Degree (n)', fontsize=10)
ax2.set_ylabel('Number of Configurations', fontsize=10)
ax2.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# 3. Heatmap of number of GPUs allocated (y)
y_matrix = np.zeros((data.J, data.K))
for (j, k), num_gpus in y_vals.items():
    y_matrix[j, k] = num_gpus

ax3 = axes[1, 0]
sns.heatmap(y_matrix, annot=True, fmt='.0f', cmap='Blues',
            xticklabels=data.gpu_tiers, yticklabels=data.model_names,
            cbar_kws={'label': 'Number of GPUs (y)'}, ax=ax3)
ax3.set_title('Number of GPUs Allocated per Model-GPU Configuration', fontsize=12, fontweight='bold')
ax3.set_xlabel('GPU Tier', fontsize=10)
ax3.set_ylabel('Model', fontsize=10)
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=8)

# 4. Detailed table showing active configurations
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

# Create table data
table_data = []
table_data.append(['Model', 'GPU Tier', 'TP Degree', '# GPUs', 'Relation'])
for (j, k) in sorted(q_vals.keys()):
    model_name = data.model_names[j]
    gpu_name = data.gpu_tiers[k]
    tp_degree = w_vals.get((j, k), 0)
    num_gpus = y_vals.get((j, k), 0)
    relation = f'{num_gpus} = {tp_degree} × 1' if tp_degree > 0 else '-'
    table_data.append([
        model_name[:15],
        gpu_name[:12],
        str(tp_degree),
        str(num_gpus),
        relation
    ])

if len(table_data) > 1:
    table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.25, 0.2, 0.15, 0.15, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')

    ax4.set_title('Active Configurations: y = n × q relationship',
                 fontsize=12, fontweight='bold', pad=20)
else:
    ax4.text(0.5, 0.5, 'No active configurations found',
            ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.savefig('/Users/jiamingcc/ASU Dropbox/Jiaming Cheng/ICC conference/tp_degree_visualization.png',
            dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to: tp_degree_visualization.png")

# Print detailed summary
print("\n" + "="*80)
print("TP DEGREE SELECTION SUMMARY")
print("="*80)

print(f"\nTP Degree Distribution:")
for n in data.TP_degrees:
    count = tp_degree_counts.get(n, 0)
    if count > 0:
        pct = count / sum(tp_degree_counts.values()) * 100
        print(f"  n = {n}: {count} configurations ({pct:.1f}%)")

print(f"\nDetailed Active Configurations:")
print(f"{'Model':<20} {'GPU Tier':<18} {'TP Degree':<12} {'# GPUs':<10} {'Verification'}")
print("-" * 80)
for (j, k) in sorted(q_vals.keys()):
    model_name = data.model_names[j]
    gpu_name = data.gpu_tiers[k]
    tp_degree = w_vals.get((j, k), 0)
    num_gpus = y_vals.get((j, k), 0)
    verify = "✓" if num_gpus == tp_degree else "✗"
    print(f"{model_name:<20} {gpu_name:<18} {tp_degree:<12} {num_gpus:<10} {verify} (y = {num_gpus}, n = {tp_degree})")

plt.show()