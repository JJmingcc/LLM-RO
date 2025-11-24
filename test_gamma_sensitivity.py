"""
Test script to analyze the impact of Gamma_d and Gamma_e on solution conservativeness
"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from RODIU_LLM import DataGenerator, LLMInferenceOptimizer, LLMInferenceData

def test_gamma_sensitivity():
    """Test how changing Gamma_d and Gamma_e affects the solution"""

    # Test different Gamma values
    gamma_values = [0, 10, 25, 50, 100, 200]

    results = []

    for gamma_d in gamma_values:
        for gamma_e in gamma_values:
            print(f"\n{'='*80}")
            print(f"Testing Gamma_d={gamma_d}, Gamma_e={gamma_e}")
            print(f"{'='*80}")

            # Generate data with specific Gamma values
            generator = DataGenerator(seed=42)
            data = generator.generate()

            # Override Gamma values
            data.Gamma_d = gamma_d
            data.Gamma_e = gamma_e

            # Solve optimization
            optimizer = LLMInferenceOptimizer(data)
            solution = optimizer.build_and_solve_optimization_problem(time_limit=300, mip_gap=0.01)

            if solution is not None:
                # Calculate metrics
                total_cost = solution['objective']
                num_gpus = sum(solution['y'].values()) if solution['y'] else 0
                unmet_demand = sum(solution['u'].values()) if solution['u'] else 0

                # Calculate robust delay penalty component
                varrho_val = optimizer.vars['varrho'].X if hasattr(optimizer.vars['varrho'], 'X') else 0

                # Calculate tau values (dual variables)
                tau_0_sum = sum(optimizer.vars['tau_0'][i].X for i in range(data.I))
                tau_1_sum = sum(optimizer.vars['tau_1'][i].X for i in range(data.I))
                tau_2_sum = sum(optimizer.vars['tau_2'][i].X for i in range(data.I))

                results.append({
                    'Gamma_d': gamma_d,
                    'Gamma_e': gamma_e,
                    'Total_Cost': total_cost,
                    'Num_GPUs': num_gpus,
                    'Unmet_Demand': unmet_demand,
                    'Robust_Delay_Penalty': varrho_val,
                    'Sum_tau_0': tau_0_sum,
                    'Sum_tau_1': tau_1_sum,
                    'Sum_tau_2': tau_2_sum,
                    'Status': solution['status']
                })

                print(f"Total Cost: ${total_cost:.2f}")
                print(f"Number of GPUs: {num_gpus}")
                print(f"Unmet Demand: {unmet_demand:.4f}")
                print(f"Robust Delay Penalty: ${varrho_val:.2f}")
                print(f"Sum of tau_0 (delay constraint duals): {tau_0_sum:.4f}")
                print(f"Sum of tau_1 (error constraint duals): {tau_1_sum:.4f}")
                print(f"Sum of tau_2 (delay objective duals): {tau_2_sum:.4f}")
            else:
                print("Optimization failed!")
                results.append({
                    'Gamma_d': gamma_d,
                    'Gamma_e': gamma_e,
                    'Total_Cost': None,
                    'Num_GPUs': None,
                    'Unmet_Demand': None,
                    'Robust_Delay_Penalty': None,
                    'Sum_tau_0': None,
                    'Sum_tau_1': None,
                    'Sum_tau_2': None,
                    'Status': 'FAILED'
                })

    # Create DataFrame and display results
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv('gamma_sensitivity_results.csv', index=False)
    print("\nResults saved to 'gamma_sensitivity_results.csv'")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Check if cost increases with Gamma
    print("\n1. Does total cost increase with Gamma?")
    for gamma_e_fixed in [0, 50, 100]:
        subset = df[df['Gamma_e'] == gamma_e_fixed].sort_values('Gamma_d')
        if len(subset) > 1:
            print(f"\n   Gamma_e = {gamma_e_fixed}:")
            for _, row in subset.iterrows():
                print(f"     Gamma_d={row['Gamma_d']:3.0f} -> Cost=${row['Total_Cost']:8.2f}")

    print("\n2. Does robust delay penalty increase with Gamma_d?")
    for gamma_e_fixed in [0, 50]:
        subset = df[df['Gamma_e'] == gamma_e_fixed].sort_values('Gamma_d')
        if len(subset) > 1:
            print(f"\n   Gamma_e = {gamma_e_fixed}:")
            for _, row in subset.iterrows():
                print(f"     Gamma_d={row['Gamma_d']:3.0f} -> Robust_Penalty=${row['Robust_Delay_Penalty']:8.2f}")

    print("\n3. Do dual variables (tau) increase with Gamma?")
    print("\n   tau_0 (delay constraint duals) vs Gamma_d:")
    for gamma_e_fixed in [0, 50]:
        subset = df[df['Gamma_e'] == gamma_e_fixed].sort_values('Gamma_d')
        if len(subset) > 1:
            print(f"\n   Gamma_e = {gamma_e_fixed}:")
            for _, row in subset.iterrows():
                print(f"     Gamma_d={row['Gamma_d']:3.0f} -> tau_0_sum={row['Sum_tau_0']:8.4f}")

    print("\n   tau_1 (error constraint duals) vs Gamma_e:")
    for gamma_d_fixed in [0, 50]:
        subset = df[df['Gamma_d'] == gamma_d_fixed].sort_values('Gamma_e')
        if len(subset) > 1:
            print(f"\n   Gamma_d = {gamma_d_fixed}:")
            for _, row in subset.iterrows():
                print(f"     Gamma_e={row['Gamma_e']:3.0f} -> tau_1_sum={row['Sum_tau_1']:8.4f}")

if __name__ == "__main__":
    test_gamma_sensitivity()