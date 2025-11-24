# Error Parameter Sensitivity Analysis

## Overview

This analysis examines how **error-related parameters** affect the total cost of the LLM inference optimization, while keeping delay impact factors **fixed**.

### What Changed from Original Analysis

**Original (2D):**
- Variable 1: Delay impact factor spacing (Δ_gamma)
- Variable 2: Error impact factor spacing (Δ_error)
- Fixed: Nothing

**New (2D):**
- **FIXED**: Delay impact factor spacing (Δ_gamma = 0.2)
- Variable 1: Error impact factor spacing (Δ_error) ∈ [0.1, 0.15, 0.2, 0.25, 0.3]
- Variable 2: Error rate threshold scaling (ε_scale) ∈ [0.5, 0.75, 1.0, 1.25, 1.5]

### Parameters Explained

1. **Δ_gamma (Fixed at 0.2)**: Controls the spacing between delay impact factor categories (low/medium/high/luxury)

2. **Δ_error (Variable)**: Controls the spacing between error impact factor categories
   - Smaller values (0.1): Categories are closer together
   - Larger values (0.3): Categories are more spread out

3. **ε_scale (Variable)**: Scales the error rate thresholds for all models
   - Base: [0.08, 0.1, 0.08, 0.1, 0.15, 0.25] for [GPT-4, Claude, Gemini, Llama, Mistral, Phi]
   - ε_scale = 0.5: Makes thresholds more strict (tighter error tolerances)
   - ε_scale = 1.5: Makes thresholds more lenient (looser error tolerances)

## Files Created

1. **error_sensitivity_analysis.py**: Main analysis script
   - Runs 25 optimization scenarios (5 × 5 parameter grid)
   - Outputs: `error_sensitivity_results.csv`

2. **error_sensitivity_visualization.py**: Visualization script
   - Creates comprehensive plots including:
     - Total cost heatmap
     - Cost component breakdown
     - Parameter sensitivity charts
     - 3D surface plot
   - Outputs: `error_sensitivity_analysis.png`, `error_sensitivity_3d.png`

3. **ERROR_SENSITIVITY_README.md**: This guide

## How to Run

### Option 1: Python Scripts

```bash
# Run analysis (takes ~2-3 hours for 25 scenarios with 300s time limit each)
python error_sensitivity_analysis.py

# Create visualizations
python error_sensitivity_visualization.py
```

### Option 2: Jupyter Notebook

Copy and paste the following code blocks into your notebook:

#### Cell 1: Run Analysis

```python
# Import the analysis module
import sys
sys.path.append('.')
from error_sensitivity_analysis import run_error_sensitivity_analysis

# Run the analysis
df_results, base_data = run_error_sensitivity_analysis()

# Save results
df_results.to_csv('error_sensitivity_results.csv', index=False)
print("\\nResults saved to 'error_sensitivity_results.csv'")
```

#### Cell 2: Create Visualizations

```python
# Import visualization module
from error_sensitivity_visualization import (
    visualize_error_sensitivity_results,
    create_3d_surface_plot
)

# Create comprehensive visualizations
visualize_error_sensitivity_results(df_results, save_figures=True)

# Create 3D surface plot
create_3d_surface_plot(df_results, save_figure=True)
```

## Expected Results

The analysis will produce:

1. **CSV File** (`error_sensitivity_results.csv`) with columns:
   - `delta_gamma`: Fixed at 0.2
   - `delta_error`: Error impact spacing (0.1-0.3)
   - `epsilon_scale`: Error threshold scale (0.5-1.5)
   - `total_cost`: Total optimization cost
   - `inference_cost`, `storage_cost`, `rental_cost`, `penalty_cost`: Cost breakdowns
   - `solve_time`: Optimization solve time
   - `status`: Optimization status (OPTIMAL/TIME_LIMIT/FAILED)

2. **Visualizations**:
   - **Heatmap**: Shows how total cost varies across the 2D parameter space
   - **Line plots**: Show cost trends for each parameter
   - **Bar charts**: Compare cost sensitivities
   - **3D surface**: Interactive view of the cost landscape

## Key Questions This Analysis Answers

1. **How does error impact spacing affect total cost?**
   - Does spreading error categories further apart increase or decrease costs?

2. **How do error rate thresholds affect total cost?**
   - Are stricter error requirements (lower ε_scale) significantly more expensive?

3. **Is there an interaction effect?**
   - Do certain combinations of Δ_error and ε_scale produce unexpectedly high/low costs?

4. **Which parameter has more influence?**
   - Does Δ_error or ε_scale have a larger impact on total cost?

5. **What is the optimal configuration?**
   - Which combination minimizes total cost?

## Customization

To modify the analysis parameters, edit `error_sensitivity_analysis.py`:

```python
# Line 233-235: Change these values
DELTA_GAMMA_FIXED = 0.2  # Try different fixed values
delta_error_values = [0.1, 0.15, 0.2, 0.25, 0.3]  # Adjust range/granularity
epsilon_scale_values = [0.5, 0.75, 1.0, 1.25, 1.5]  # Adjust range/granularity
```

To adjust optimization parameters:

```python
# Line 258-259: Change solver settings
time_limit=300,  # Increase for better solutions
mip_gap=0.01     # Decrease for tighter optimality
```

## Time Estimates

- **Single scenario**: ~5-10 minutes (300s time limit + overhead)
- **Full analysis (25 scenarios)**: ~2-3 hours
- **Visualization**: <1 minute

## Notes

- The analysis uses the same base parameter configuration as your main experiments
- Random seeds are fixed for reproducibility (seed=42 for gamma, seed=43 for error)
- Failed scenarios are excluded from visualizations but retained in the CSV
- The 3D plot can be rotated interactively if running in Jupyter notebook

## Troubleshooting

**Issue**: Analysis takes too long
- **Solution**: Reduce `time_limit` (e.g., 120s instead of 300s) or use fewer parameter values

**Issue**: Many failed scenarios
- **Solution**: Check parameter ranges - extreme values may create infeasible problems

**Issue**: Visualizations don't show
- **Solution**: Ensure matplotlib backend is properly configured: `%matplotlib inline` in Jupyter

## Next Steps

After running the analysis, you can:

1. Compare results with the original 2D analysis (gamma vs error spacing)
2. Identify which parameter (Δ_error or ε_scale) has more impact
3. Use insights to guide parameter selection for production scenarios
4. Perform additional targeted analyses on interesting regions of the parameter space