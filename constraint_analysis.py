"""
Constraint Tightness Analysis

This script analyzes which constraints are binding (tight) and limiting the solution.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import sys
sys.path.append('/Users/jiamingcc/ASU Dropbox/Jiaming Cheng/ICC conference')

from robust_llm_optimization import RobustDataGenerator, RobustLLMInferenceOptimizer, RobustLLMInferenceData
import copy


def analyze_constraint_slack(optimizer, data):
    """Analyze slack/tightness of all constraints"""

    print("\n" + "="*80)
    print("CONSTRAINT SLACK ANALYSIS")
    print("="*80)
    print("\nSlack = 0 means constraint is TIGHT (binding)")
    print("Slack > 0 means constraint has room (not binding)")

    # Get all constraints
    model = optimizer.model

    # Extract variable values
    x = {(i,j,k): optimizer.vars['x'][i,j,k].X for i in range(data.I) for j in range(data.J) for k in range(data.K)}
    y = {(j,k): optimizer.vars['y'][j,k].X for j in range(data.J) for k in range(data.K)}
    z = {(i,j,k): optimizer.vars['z'][i,j,k].X for i in range(data.I) for j in range(data.J) for k in range(data.K)}
    u = {(i,k): optimizer.vars['u'][i,k].X for i in range(data.I) for k in range(data.K)}
    w = {(j,k,n): optimizer.vars['w'][j,k,n].X for j in range(data.J) for k in range(data.K) for n in data.TP_degrees}
    v = {(i,j,k,n): optimizer.vars['v'][i,j,k,n].X for i in range(data.I) for j in range(data.J) for k in range(data.K) for n in data.TP_degrees}
    q = {(j,k): optimizer.vars['q'][j,k].X for j in range(data.J) for k in range(data.K)}

    # 1. MEMORY CONSTRAINTS
    print("\n" + "-"*80)
    print("1. MEMORY CAPACITY CONSTRAINTS")
    print("-"*80)
    print(f"{'Model':<25} {'GPU':<20} {'Used (GB)':>12} {'Capacity':>12} {'Slack':>12} {'Util%':>10}")
    print("-"*80)

    memory_tight_count = 0
    memory_constraints = []

    for j in range(data.J):
        for k in range(data.K):
            if y[j,k] > 0.01:  # Only check if GPUs are allocated
                # Calculate memory usage
                model_weight = sum((data.B[j] / n) * w[j,k,n] for n in data.TP_degrees)
                kv_cache = sum((data.beta[j] / n) * (data.h[i] + data.f[i]) * data.lambda_i[i] *
                              data.T_res[i,j,k] * v[i,j,k,n]
                              for i in range(data.I) for n in data.TP_degrees)

                total_used = model_weight + kv_cache
                capacity = data.C_gpu[k] * q[j,k]
                slack = capacity - total_used
                utilization = (total_used / capacity * 100) if capacity > 0 else 0

                memory_constraints.append({
                    'model': data.model_names[j],
                    'gpu': data.gpu_tiers[k],
                    'used': total_used,
                    'capacity': capacity,
                    'slack': slack,
                    'util': utilization
                })

                if slack < 0.1:  # Tight if slack < 0.1 GB
                    memory_tight_count += 1
                    status = "TIGHT!"
                else:
                    status = ""

                print(f"{data.model_names[j]:<25} {data.gpu_tiers[k]:<20} {total_used:>12.2f} {capacity:>12.2f} {slack:>12.2f} {utilization:>9.1f}% {status}")

    print(f"\nMemory constraints that are TIGHT: {memory_tight_count}")

    # 2. COMPUTE CAPACITY CONSTRAINTS
    print("\n" + "-"*80)
    print("2. COMPUTE CAPACITY CONSTRAINTS")
    print("-"*80)
    print(f"{'Model':<25} {'GPU':<20} {'Used (TFLOP)':>15} {'Capacity':>12} {'Slack':>12} {'Util%':>10}")
    print("-"*80)

    compute_tight_count = 0
    compute_constraints = []

    for j in range(data.J):
        for k in range(data.K):
            if y[j,k] > 0.01:
                # Calculate compute usage
                compute_used = sum(data.alpha[i] * (data.f[i] + data.h[i]) * data.lambda_i[i] * x[i,j,k]
                                  for i in range(data.I))
                capacity = data.P_gpu[k] * y[j,k]
                slack = capacity - compute_used
                utilization = (compute_used / capacity * 100) if capacity > 0 else 0

                compute_constraints.append({
                    'model': data.model_names[j],
                    'gpu': data.gpu_tiers[k],
                    'used': compute_used,
                    'capacity': capacity,
                    'slack': slack,
                    'util': utilization
                })

                if slack < 1.0:  # Tight if slack < 1 TFLOP
                    compute_tight_count += 1
                    status = "TIGHT!"
                else:
                    status = ""

                print(f"{data.model_names[j]:<25} {data.gpu_tiers[k]:<20} {compute_used:>15.2f} {capacity:>12.2f} {slack:>12.2f} {utilization:>9.1f}% {status}")

    print(f"\nCompute constraints that are TIGHT: {compute_tight_count}")

    # 3. DELAY CONSTRAINTS
    print("\n" + "-"*80)
    print("3. ROBUST DELAY CONSTRAINTS")
    print("-"*80)
    print(f"{'Query Type':<20} {'Delay (ms)':>15} {'Threshold':>12} {'Slack':>12} {'Util%':>10}")
    print("-"*80)

    delay_tight_count = 0

    for i in range(data.I):
        # Calculate total delay for this query type
        total_delay = sum((data.d_bar[i] + data.d_hat[i]) * (data.h[i] + data.f[i]) *
                         data.lambda_i[i] * x[i,j,k]
                         for j in range(data.J) for k in range(data.K))
        threshold = data.Delta_i[i]
        slack = threshold - total_delay
        utilization = (total_delay / threshold * 100) if threshold > 0 else 0

        if slack < 10:  # Tight if slack < 10ms
            delay_tight_count += 1
            status = "TIGHT!"
        else:
            status = ""

        print(f"{data.query_types[i]:<20} {total_delay:>15.2f} {threshold:>12.2f} {slack:>12.2f} {utilization:>9.1f}% {status}")

    print(f"\nDelay constraints that are TIGHT: {delay_tight_count}")

    # 4. ERROR RATE CONSTRAINTS
    print("\n" + "-"*80)
    print("4. ROBUST ERROR RATE CONSTRAINTS")
    print("-"*80)
    print(f"{'Query Type':<20} {'Error Rate':>15} {'Threshold':>12} {'Slack':>12} {'Util%':>10}")
    print("-"*80)

    error_tight_count = 0

    for i in range(data.I):
        # Calculate total error rate for this query type
        total_error = sum((data.e_bar[i] + data.e_hat[i]) * x[i,j,k]
                         for j in range(data.J) for k in range(data.K))
        threshold = data.epsilon[i]
        slack = threshold - total_error
        utilization = (total_error / threshold * 100) if threshold > 0 else 0

        if slack < 0.001:  # Tight if slack < 0.1%
            error_tight_count += 1
            status = "TIGHT!"
        else:
            status = ""

        print(f"{data.query_types[i]:<20} {total_error:>15.4f} {threshold:>12.4f} {slack:>12.4f} {utilization:>9.1f}% {status}")

    print(f"\nError rate constraints that are TIGHT: {error_tight_count}")

    # 5. BUDGET CONSTRAINT
    print("\n" + "-"*80)
    print("5. BUDGET CONSTRAINT")
    print("-"*80)

    # Calculate total budget used
    budget_used = data.Delta_T * sum(
        data.p_c[k] * y[j,k] +
        data.p_s * (data.B[j] * z[i,j,k] + data.theta[i] * (data.h[i] + data.f[i]) * data.lambda_i[i] * x[i,j,k])
        for i in range(data.I) for j in range(data.J) for k in range(data.K)
    )
    budget_capacity = data.delta
    budget_slack = budget_capacity - budget_used
    budget_util = (budget_used / budget_capacity * 100) if budget_capacity > 0 else 0

    print(f"Budget Used:      ${budget_used:>15.2f}")
    print(f"Budget Available: ${budget_capacity:>15.2f}")
    print(f"Slack:            ${budget_slack:>15.2f}")
    print(f"Utilization:      {budget_util:>14.1f}%")

    if budget_slack < 100:
        print("Status: TIGHT!")

    # 6. STORAGE CONSTRAINT
    print("\n" + "-"*80)
    print("6. STORAGE CONSTRAINT")
    print("-"*80)

    storage_used = sum(data.B[j] * z[i,j,k] for i in range(data.I) for j in range(data.J) for k in range(data.K))
    storage_capacity = data.C_storage
    storage_slack = storage_capacity - storage_used
    storage_util = (storage_used / storage_capacity * 100) if storage_capacity > 0 else 0

    print(f"Storage Used:      {storage_used:>15.2f} GB")
    print(f"Storage Available: {storage_capacity:>15.2f} GB")
    print(f"Slack:             {storage_slack:>15.2f} GB")
    print(f"Utilization:       {storage_util:>14.1f}%")

    if storage_slack < 10:
        print("Status: TIGHT!")

    # SUMMARY
    print("\n" + "="*80)
    print("CONSTRAINT TIGHTNESS SUMMARY")
    print("="*80)
    print(f"Memory constraints (tight):       {memory_tight_count}")
    print(f"Compute constraints (tight):      {compute_tight_count}")
    print(f"Delay constraints (tight):        {delay_tight_count}")
    print(f"Error rate constraints (tight):   {error_tight_count}")
    print(f"Budget constraint (tight):        {'YES' if budget_slack < 100 else 'NO'}")
    print(f"Storage constraint (tight):       {'YES' if storage_slack < 10 else 'NO'}")

    print("\n" + "="*80)
    print("BOTTLENECK IDENTIFICATION")
    print("="*80)

    # Identify the main bottleneck
    if error_tight_count > 0:
        print("⚠️  PRIMARY BOTTLENECK: Error Rate Constraints")
        print("    → Robust error rate constraints are limiting workload allocation")
        print("    → Even with more GPUs, cannot allocate more workload due to error thresholds")
        print("    → Solution: Relax epsilon (error rate thresholds) or reduce e_hat (uncertainty)")

    if delay_tight_count > 0:
        print("⚠️  PRIMARY BOTTLENECK: Delay Constraints")
        print("    → Robust delay constraints are limiting workload allocation")
        print("    → Cannot serve more queries without violating delay SLAs")
        print("    → Solution: Relax Delta_i (delay thresholds) or reduce d_hat (uncertainty)")

    if compute_tight_count > 0:
        print("⚠️  BOTTLENECK: Compute Capacity")
        print("    → GPUs are running at full compute capacity")
        print("    → Need more or faster GPUs to handle additional workload")

    if memory_tight_count > 0:
        print("⚠️  BOTTLENECK: Memory Capacity")
        print("    → GPUs are running at full memory capacity")
        print("    → Consider higher memory GPUs or higher TP degree")

    if budget_slack < 100:
        print("⚠️  BOTTLENECK: Budget")
        print("    → Budget is fully utilized")
        print("    → Need higher budget to rent more/better GPUs")

    return {
        'memory_tight': memory_tight_count,
        'compute_tight': compute_tight_count,
        'delay_tight': delay_tight_count,
        'error_tight': error_tight_count,
        'budget_tight': budget_slack < 100,
        'storage_tight': storage_slack < 10
    }


# ============================================================================
# ANALYZE BASELINE SCENARIO
# ============================================================================
print("="*80)
print("ANALYZING BASELINE SCENARIO")
print("="*80)

generator1 = RobustDataGenerator(seed=42, delay_uncertainty=0.2, error_uncertainty=0.3)
data1 = generator1.generate()

optimizer1 = RobustLLMInferenceOptimizer(data1)
optimizer1.build_model()
solution1 = optimizer1.solve(time_limit=300, mip_gap=0.01)

bottlenecks1 = analyze_constraint_slack(optimizer1, data1)


# ============================================================================
# ANALYZE HIGH PENALTY SCENARIO
# ============================================================================
print("\n\n" + "="*80)
print("ANALYZING HIGH PENALTY SCENARIO (50x penalty + 10x budget + 5x demand)")
print("="*80)

data2 = copy.deepcopy(data1)
data2.delta = data1.delta * 10
data2.phi = data1.phi * 50
data2.lambda_i = data1.lambda_i * 5

optimizer2 = RobustLLMInferenceOptimizer(data2)
optimizer2.build_model()
solution2 = optimizer2.solve(time_limit=300, mip_gap=0.01)

bottlenecks2 = analyze_constraint_slack(optimizer2, data2)


# ============================================================================
# TEST: RELAX ERROR CONSTRAINTS
# ============================================================================
print("\n\n" + "="*80)
print("TEST: RELAXING ERROR RATE CONSTRAINTS (2x epsilon)")
print("="*80)

data3 = copy.deepcopy(data2)  # Start from high penalty scenario
data3.epsilon = data1.epsilon * 2  # Double the error rate thresholds

print(f"\nOriginal Error Thresholds: {data1.epsilon}")
print(f"Relaxed Error Thresholds:  {data3.epsilon}")

optimizer3 = RobustLLMInferenceOptimizer(data3)
optimizer3.build_model()
solution3 = optimizer3.solve(time_limit=300, mip_gap=0.01)

print(f"\nServed Percentage Comparison:")
u1 = sum(sum(solution2['u'].get((i,k), 0) for k in range(data2.K)) * data2.lambda_i[i] for i in range(data2.I))
served1 = (sum(data2.lambda_i) - u1) / sum(data2.lambda_i)

u3 = sum(sum(solution3['u'].get((i,k), 0) for k in range(data3.K)) * data3.lambda_i[i] for i in range(data3.I))
served3 = (sum(data3.lambda_i) - u3) / sum(data3.lambda_i)

print(f"  Original (tight error constraints): {served1:.1%}")
print(f"  Relaxed (2x error thresholds):      {served3:.1%}")
print(f"  Improvement:                        {(served3-served1)*100:.1f} percentage points")

bottlenecks3 = analyze_constraint_slack(optimizer3, data3)