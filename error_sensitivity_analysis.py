# ============================================================================
# 2D SENSITIVITY ANALYSIS: Error Impact Factor Spacing vs Error Rate Threshold
# ============================================================================
# This analysis tests how error-related parameters affect the optimization results
# while keeping delay impact factors FIXED.
#
# Fixed: Delta_impact_gamma = 0.2 (delay impact spacing)
#        Low impact range = (0.05, 0.1) for both gamma and error
# Variable: Delta_impact_error ∈ [0.1, 0.15, 0.2, 0.25, 0.3]
#           Epsilon_scale ∈ [0.5, 0.75, 1.0, 1.25, 1.5] (error threshold scaling)
# Total scenarios: 5 x 5 = 25
# ============================================================================

import numpy as np
import pandas as pd
from typing import Dict, Any
import importlib

# Reload modules to get latest changes
import parameter_setup
import RDDU_LLM_inference_opt
importlib.reload(parameter_setup)
importlib.reload(RDDU_LLM_inference_opt)

from parameter_setup import ParameterGenerator, LLMInferenceData
from RDDU_LLM_inference_opt import LLMInferenceOptimizer, LLMInferenceData as RDDUData

def generate_custom_impact_ranges(delta_impact: float) -> Dict[str, tuple]:
    """
    Generate impact ranges based on delta_impact spacing.

    Args:
        delta_impact: Spacing between adjacent category centers

    Returns:
        Dictionary of impact ranges for each category
    """
    low_center = 0.075  # Fixed: (0.05, 0.1)
    range_width = 0.05  # Width of each range
    half_width = range_width / 2

    impact_ranges = {
        'low': (0.05, 0.1),
        'medium': (low_center + delta_impact - half_width,
                   low_center + delta_impact + half_width),
        'high': (low_center + 2*delta_impact - half_width,
                 low_center + 2*delta_impact + half_width),
        'luxury': (low_center + 3*delta_impact - half_width,
                   low_center + 3*delta_impact + half_width)
    }

    return impact_ranges

def generate_custom_gamma_impact(J: int,
                                delta_impact: float,
                                seed: int = 42) -> np.ndarray:
    """Generate delay impact factors with custom spacing."""
    np.random.seed(seed)
    gamma_impact = np.zeros(J)
    impact_ranges = generate_custom_impact_ranges(delta_impact)

    # Model type assignments (same as original)
    assignments = {
        0: 'luxury',   # GPT-4
        1: 'high',     # Claude
        2: 'high',     # Gemini
        3: 'medium',   # Llama
        4: 'low',      # Mistral
        5: 'low'       # Phi
    }

    for j in range(J):
        category = assignments[j]
        low, high = impact_ranges[category]
        gamma_impact[j] = np.random.uniform(low, high)

    return gamma_impact

def generate_custom_error_impact(J: int,
                                 delta_impact: float,
                                 seed: int = 43) -> np.ndarray:
    """Generate error impact factors with custom spacing."""
    np.random.seed(seed)
    eta = np.zeros(J)
    impact_ranges = generate_custom_impact_ranges(delta_impact)

    # Model type assignments (same as original)
    assignments = {
        0: 'low',      # GPT-4
        1: 'low',      # Claude
        2: 'low',      # Gemini
        3: 'medium',   # Llama
        4: 'high',     # Mistral
        5: 'luxury'    # Phi
    }

    for j in range(J):
        category = assignments[j]
        low, high = impact_ranges[category]
        eta[j] = np.random.uniform(low, high)

    return eta

def run_error_sensitivity_scenario(base_data: LLMInferenceData,
                                   delta_gamma: float,
                                   delta_error: float,
                                   epsilon_scale: float,
                                   time_limit: int = 300,
                                   mip_gap: float = 0.01) -> Dict[str, Any]:
    """
    Run a single scenario with specified parameters.

    Args:
        base_data: Base parameter configuration
        delta_gamma: Spacing for delay impact factors (FIXED)
        delta_error: Spacing for error impact factors (VARIABLE)
        epsilon_scale: Scaling factor for error rate thresholds (VARIABLE)
        time_limit: Optimization time limit in seconds
        mip_gap: MIP gap tolerance

    Returns:
        Dictionary containing results
    """
    # Generate custom impact factors
    gamma_impact = generate_custom_gamma_impact(base_data.J, delta_gamma)
    eta = generate_custom_error_impact(base_data.J, delta_error)

    # Scale error rate thresholds
    epsilon_scaled = base_data.epsilon * epsilon_scale

    # Create modified data object
    modified_data = RDDUData(
        # Basic parameters
        J=base_data.J, I=base_data.I, T=base_data.T,

        # Model parameters with custom impact factors
        gamma_impact=gamma_impact,  # Custom delay impact
        eta=eta,                     # Custom error impact

        # Use original parameters for others
        mu=base_data.mu, p_u=base_data.p_u, p_c=base_data.p_c,
        rho=base_data.rho, sigma=base_data.sigma, tau=base_data.tau,

        # Demand and penalty
        lambda_demand=base_data.lambda_demand, phi=base_data.phi,

        # Thresholds with scaled epsilon
        delta=base_data.delta, Delta_T=base_data.Delta_T,
        Delta_i=base_data.Delta_i, epsilon=epsilon_scaled,  # SCALED!
        C_storage=base_data.C_storage,

        # Resource parameters
        T_res=base_data.T_res, BigM=base_data.BigM
    )

    # Solve optimization
    try:
        optimizer = LLMInferenceOptimizer(modified_data)
        solution = optimizer.build_and_solve_optimization_problem(
            time_limit=time_limit, mip_gap=mip_gap
        )

        if solution is None:
            return {
                'delta_gamma': delta_gamma,
                'delta_error': delta_error,
                'epsilon_scale': epsilon_scale,
                'status': 'FAILED',
                'total_cost': None,
                'inference_cost': None,
                'storage_cost': None,
                'rental_cost': None,
                'penalty_cost': None,
                'solve_time': None
            }

        return {
            'delta_gamma': delta_gamma,
            'delta_error': delta_error,
            'epsilon_scale': epsilon_scale,
            'status': solution['status'],
            'total_cost': solution['total_cost'],
            'inference_cost': solution.get('inference_cost', None),
            'storage_cost': solution.get('storage_cost', None),
            'rental_cost': solution.get('rental_cost', None),
            'penalty_cost': solution.get('penalty_cost', None),
            'solve_time': solution.get('solve_time', None),
            'gamma_impact_stats': {
                'mean': gamma_impact.mean(),
                'std': gamma_impact.std(),
                'min': gamma_impact.min(),
                'max': gamma_impact.max()
            },
            'eta_stats': {
                'mean': eta.mean(),
                'std': eta.std(),
                'min': eta.min(),
                'max': eta.max()
            },
            'epsilon_scaled': epsilon_scaled.tolist()
        }

    except Exception as e:
        print(f"Error in scenario: {str(e)}")
        return {
            'delta_gamma': delta_gamma,
            'delta_error': delta_error,
            'epsilon_scale': epsilon_scale,
            'status': 'ERROR',
            'total_cost': None,
            'error_message': str(e)
        }

def run_error_sensitivity_analysis():
    """
    Run the complete 2D sensitivity analysis for error parameters.
    """
    print("="*80)
    print("2D SENSITIVITY ANALYSIS: Error Impact Spacing vs Error Rate Threshold")
    print("="*80)

    # Generate base parameters
    print("\nGenerating base parameters...")
    param_gen = ParameterGenerator(num_models=6, num_instances=4, time_periods=12)
    base_data = param_gen.generate_parameters()

    print(f"  Models: {base_data.J}")
    print(f"  Instances: {base_data.I}")
    print(f"  Time periods: {base_data.T}")
    print(f"  Base epsilon (error thresholds): {base_data.epsilon}")

    # Define parameter ranges
    DELTA_GAMMA_FIXED = 0.2  # Fixed delay impact spacing
    delta_error_values = [0.1, 0.15, 0.2, 0.25, 0.3]
    epsilon_scale_values = [0.5, 0.75, 1.0, 1.25, 1.5]

    print(f"\nParameter ranges:")
    print(f"  Delta_gamma (FIXED): {DELTA_GAMMA_FIXED}")
    print(f"  Delta_error: {delta_error_values}")
    print(f"  Epsilon_scale: {epsilon_scale_values}")
    print(f"  Total scenarios: {len(delta_error_values) * len(epsilon_scale_values)}")

    # Run analysis
    results = []
    counter = 0
    total = len(delta_error_values) * len(epsilon_scale_values)

    for delta_error in delta_error_values:
        for epsilon_scale in epsilon_scale_values:
            counter += 1
            print(f"\n[{counter}/{total}] Delta_error={delta_error:.2f}, Epsilon_scale={epsilon_scale:.2f}")

            result = run_error_sensitivity_scenario(
                base_data=base_data,
                delta_gamma=DELTA_GAMMA_FIXED,
                delta_error=delta_error,
                epsilon_scale=epsilon_scale,
                time_limit=300,
                mip_gap=0.01
            )

            results.append(result)

            if result['status'] in ['OPTIMAL', 'TIME_LIMIT']:
                print(f"  Status: {result['status']}")
                print(f"  Total cost: ${result['total_cost']:,.2f}")
                print(f"  Solve time: {result['solve_time']:.2f}s")
            else:
                print(f"  Status: {result['status']}")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nSuccessful scenarios: {len(df_results[df_results['status'].isin(['OPTIMAL', 'TIME_LIMIT'])])}/{total}")

    return df_results, base_data

# Run the analysis
if __name__ == "__main__":
    df_results, base_data = run_error_sensitivity_analysis()

    # Save results
    df_results.to_csv('error_sensitivity_results.csv', index=False)
    print("\nResults saved to 'error_sensitivity_results.csv'")