"""
Analysis of why error rate constraints are not restrictive
"""
from parameter_setup import ParameterGenerator
import numpy as np

gen = ParameterGenerator(seed=42)

# Get parameters
I, J, K, N, TP_degrees = gen.get_problem_dimensions()
query_types = gen.get_query_types()
h = gen.get_input_token_lengths()
f = gen.get_output_token_lengths()
Delta_i = np.array([1000, 1500, 800, 2000, 4000, 7000])
epsilon = np.array([0.08, 0.1, 0.08, 0.1, 0.15, 0.25])

# Generate error and delay parameters
d_bar = gen.generate_processing_delays(I, J, K)
e_bar = gen.generate_error_rates(I, J, K)
d_bar_copy, d_hat = gen.generate_uncertainty_delays(d_bar)
e_bar_copy, e_hat = gen.generate_uncertainty_error_rates(e_bar)

print('='*100)
print('CONSTRAINT FORMULATION ANALYSIS')
print('='*100)

print('\n' + '='*100)
print('KEY DIFFERENCE IN CONSTRAINT FORMULATION:')
print('='*100)
print('\nDELAY CONSTRAINT:')
print('  LHS: Σ d_bar[i,j,k] * (h[i] + f[i]) * x[i,j,k] + Γ_d*τ + Σσ')
print('  RHS: Δ_i[i]  (absolute threshold in milliseconds)')
print('  Note: LHS is TOTAL DELAY for query type i')
print('')
print('ERROR CONSTRAINT:')
print('  LHS: Σ e_bar[i,j,k] * x[i,j,k] + Γ_e*τ + Σσ')
print('  RHS: ε[i]  (threshold as fraction, e.g., 0.08 = 8%)')
print('  Note: LHS is WEIGHTED AVERAGE ERROR RATE (since Σx[i,j,k] ≤ 1)')
print('')
print('='*100)

print('\n' + '='*100)
print('ANALYSIS BY QUERY TYPE:')
print('='*100)

for i, task in enumerate(query_types):
    print(f'\n{"-"*100}')
    print(f'{task.upper()}')
    print(f'{"-"*100}')

    # Error analysis
    print(f'\n[ERROR CONSTRAINT]')
    print(f'  Threshold: ε[{i}] = {epsilon[i]:.4f}')

    # Find configurations that satisfy error threshold
    e_worst = e_bar[i] + e_hat[i]
    configs_satisfy_error = np.sum(e_worst <= epsilon[i])
    total_configs = J * K

    print(f'  Nominal error range: [{e_bar[i].min():.4f}, {e_bar[i].max():.4f}]')
    print(f'  Worst-case error range: [{e_worst.min():.4f}, {e_worst.max():.4f}]')
    print(f'  Configs satisfying threshold: {configs_satisfy_error}/{total_configs} ({100*configs_satisfy_error/total_configs:.1f}%)')

    if configs_satisfy_error >= total_configs * 0.5:
        print(f'  ⚠️  MANY CONFIGS SATISFY ERROR THRESHOLD → Constraint is NOT RESTRICTIVE!')

    # Delay analysis
    print(f'\n[DELAY CONSTRAINT]')
    print(f'  Threshold: Δ[{i}] = {Delta_i[i]:.1f} ms')

    total_tokens = h[i] + f[i]
    d_worst = d_bar[i] + d_hat[i]
    total_delay_worst = d_worst * total_tokens  # Total delay per query

    configs_satisfy_delay = np.sum(total_delay_worst <= Delta_i[i])

    print(f'  Total tokens: {total_tokens}')
    print(f'  Delay per token range: [{d_worst.min():.4f}, {d_worst.max():.4f}] ms/token')
    print(f'  Total delay range: [{total_delay_worst.min():.1f}, {total_delay_worst.max():.1f}] ms')
    print(f'  Configs satisfying threshold: {configs_satisfy_delay}/{total_configs} ({100*configs_satisfy_delay/total_configs:.1f}%)')

    if configs_satisfy_delay < total_configs * 0.5:
        print(f'  ⚠️  MANY CONFIGS VIOLATE DELAY THRESHOLD → Constraint is VERY RESTRICTIVE!')

print('\n' + '='*100)
print('ROOT CAUSE SUMMARY:')
print('='*100)
print('''
Why error constraints are not restrictive:

1. ERROR CONSTRAINT FORMULATION:
   - Constraint: Σ e_bar[i,j,k] * x[i,j,k] ≤ ε[i]
   - Since x[i,j,k] are fractions summing to ≤1, this is a WEIGHTED AVERAGE
   - The optimizer can choose configs with low error rates
   - Many configs already have error rates below thresholds!

2. DELAY CONSTRAINT FORMULATION:
   - Constraint: Σ d_bar[i,j,k] * (h[i] + f[i]) * x[i,j,k] ≤ Δ[i]
   - This multiplies by total tokens, making it an ABSOLUTE CONSTRAINT
   - Even with TP parallelism, many configs violate the threshold
   - Forces optimizer to carefully select fast GPU configs

3. THE KEY INSIGHT:
   - Error thresholds (e.g., 0.08 = 8%) are GENEROUS relative to actual error rates
   - The optimizer can almost always find a combination that satisfies error constraints
   - Changing ε from 0.08 to 0.10 doesn't matter if most configs are already < 0.08!

4. DELAY THRESHOLDS ARE TIGHT:
   - Many task-model-GPU combinations exceed delay thresholds
   - Optimizer is FORCED to pick specific fast configs
   - This drives up GPU costs (need expensive H100s instead of cheap A6000s)
   - Changing Δ significantly impacts which configs are feasible → affects cost!

RECOMMENDATION:
To make error constraints more restrictive:
1. TIGHTEN error thresholds (e.g., ε = 0.03 instead of 0.08)
2. OR increase base error rates in parameter generation
3. OR change constraint formulation to make errors more expensive
''')
print('='*100)