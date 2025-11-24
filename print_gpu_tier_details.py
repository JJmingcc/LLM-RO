import numpy as np
from RDDU_LLM_inference_opt_refactored import DataGenerator

def impact_level(value: float) -> str:
    """Convert impact value to qualitative level"""
    if value < 0.3:
        return "Low"
    elif value < 0.5:
        return "Medium"
    elif value < 0.7:
        return "High"
    else:
        return "Luxury"

def print_gpu_tier_details(data):
    """Print detailed GPU tier information with error and delay impacts"""
    print("\n" + "="*90)
    print(f"DETAILED GPU TIER INFORMATION (Total K={data.K})")
    print("="*90)

    for k in range(data.K):
        print(f"\n{'='*90}")
        print(f"GPU Tier {k}: {data.gpu_tiers[k]}")
        print(f"  Compute Power: {data.P_gpu[k]:.1f} TFLOPS")
        print(f"  Memory: {data.C_gpu[k]} GB")
        print(f"  Cost: ${data.p_c[k]:.3f}/hr")
        print(f"{'='*90}")

        print(f"\n  {'Model':<25s} {'Delay Impact':<20s} {'Error Impact':<20s}")
        print(f"  {'-'*65}")
        for j in range(data.J):
            delay_impact = data.gamma_impact[j, k]
            error_impact_val = data.error_impact[j, k]
            print(f"  {data.model_names[j]:<25s} {delay_impact:>6.4f} ({impact_level(delay_impact):<6s})      {error_impact_val:>6.4f} ({impact_level(error_impact_val):<6s})")

        # Summary statistics for this GPU tier
        avg_delay = np.mean(data.gamma_impact[:, k])
        avg_error = np.mean(data.error_impact[:, k])
        min_delay = np.min(data.gamma_impact[:, k])
        max_delay = np.max(data.gamma_impact[:, k])
        min_error = np.min(data.error_impact[:, k])
        max_error = np.max(data.error_impact[:, k])

        print(f"  {'-'*65}")
        print(f"  {'Average':<25s} {avg_delay:>6.4f} ({impact_level(avg_delay):<6s})      {avg_error:>6.4f} ({impact_level(avg_error):<6s})")
        print(f"  {'Min':<25s} {min_delay:>6.4f} ({impact_level(min_delay):<6s})      {min_error:>6.4f} ({impact_level(min_error):<6s})")
        print(f"  {'Max':<25s} {max_delay:>6.4f} ({impact_level(max_delay):<6s})      {max_error:>6.4f} ({impact_level(max_error):<6s})")

    # Overall summary across all GPU tiers
    print(f"\n{'='*90}")
    print("OVERALL SUMMARY ACROSS ALL GPU TIERS")
    print(f"{'='*90}")

    overall_avg_delay = np.mean(data.gamma_impact)
    overall_avg_error = np.mean(data.error_impact)
    overall_std_delay = np.std(data.gamma_impact)
    overall_std_error = np.std(data.error_impact)

    print(f"\nDelay Impact Statistics:")
    print(f"  Mean: {overall_avg_delay:.4f} ({impact_level(overall_avg_delay)})")
    print(f"  Std Dev: {overall_std_delay:.4f}")
    print(f"  Min: {np.min(data.gamma_impact):.4f}")
    print(f"  Max: {np.max(data.gamma_impact):.4f}")

    print(f"\nError Impact Statistics:")
    print(f"  Mean: {overall_avg_error:.4f} ({impact_level(overall_avg_error)})")
    print(f"  Std Dev: {overall_std_error:.4f}")
    print(f"  Min: {np.min(data.error_impact):.4f}")
    print(f"  Max: {np.max(data.error_impact):.4f}")


if __name__ == "__main__":
    print("Generating data and printing GPU tier details...")

    # Generate data
    generator = DataGenerator(seed=42)
    data = generator.generate()

    # Print detailed GPU tier information
    print_gpu_tier_details(data)