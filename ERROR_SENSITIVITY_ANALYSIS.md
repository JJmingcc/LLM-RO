# Why Error Rate Thresholds Don't Impact Total Cost in Sensitivity Analysis

## Executive Summary

**Finding**: Changing error rate thresholds (ε) has minimal impact on total cost, while delay thresholds (Δ) significantly affect cost.

**Root Cause**: Error constraints are NOT BINDING because:
1. Error thresholds are too generous (83-100% of configurations already satisfy them)
2. Error constraints are formulated as weighted averages, giving optimizer flexibility
3. The optimizer can almost always find low-error configurations without cost tradeoffs

---

## Detailed Analysis

### 1. Constraint Formulation Comparison

#### ERROR RATE CONSTRAINT (RODIU_LLM.py, line 608-609):
```
Σ e_bar[i,j,k] * x[i,j,k] + Γ_e*τ + Σσ ≤ ε[i]
```

**Key characteristics:**
- LHS is a **WEIGHTED AVERAGE** error rate
- Since Σx[i,j,k] ≤ 1 (workload fractions), optimizer can blend low-error configs
- RHS is a simple fraction threshold (e.g., 0.08 = 8%)
- **Flexible**: Optimizer can choose any combination of configs that averages below threshold

#### DELAY CONSTRAINT (RODIU_LLM.py, line 602-604):
```
Σ d_bar[i,j,k] * (h[i] + f[i]) * x[i,j,k] + Γ_d*τ + Σσ ≤ Δ[i] * Σ(n*w[j,k,n])
```

**Key characteristics:**
- LHS includes multiplication by **(h[i] + f[i])** - total tokens per query
- Creates an **ABSOLUTE DELAY** constraint in milliseconds
- RHS is adjusted by TP parallelism degree
- **Rigid**: Even fast GPUs may violate threshold due to token multiplication

---

### 2. Empirical Evidence

#### Percentage of Configurations Satisfying Thresholds:

| Task          | Error Threshold | Configs Satisfying | Delay Threshold | Configs Satisfying |
|---------------|-----------------|--------------------|-----------------|--------------------|
| Summarization | 8%              | **90%** ✓          | 1000ms          | 87%                |
| Code_Gen      | 10%             | **100%** ✓         | 1500ms          | 83%                |
| Translation   | 8%              | **98%** ✓          | 800ms           | **100%** ✓         |
| Math_Solving  | 10%             | **83%** ✓          | 2000ms          | **100%** ✓         |
| Image_Gen     | 15%             | **90%** ✓          | 4000ms          | 75%                |
| Video_Gen     | 25%             | **90%** ✓          | 7000ms          | 62%                |

**Observation**:
- **ERROR**: 83-100% of configs satisfy error thresholds → Many feasible choices
- **DELAY**: Varies widely, with some tasks having only 62% satisfying → Forces specific choices

---

### 3. Why Sensitivity Analysis Shows No Impact

When you run sensitivity analysis varying `psi_error` (e.g., from 0.5× to 2×):

#### Current Error Thresholds (from parameter_setup.py:164):
```python
epsilon = [0.08, 0.10, 0.08, 0.10, 0.15, 0.25]
```

#### What Happens When You Vary psi_error:

| psi_error | Summarization Threshold | Configs Satisfying | Impact on Cost |
|-----------|-------------------------|--------------------|----------------|
| 0.5×      | 0.04                    | 48/60 (80%)        | Minimal        |
| 1.0×      | 0.08                    | 54/60 (90%)        | **Baseline**   |
| 1.5×      | 0.12                    | 60/60 (100%)       | Minimal        |
| 2.0×      | 0.16                    | 60/60 (100%)       | Minimal        |

**Problem**: Once psi_error ≥ 1.0, **ALL configurations satisfy the threshold**!
- Relaxing from 0.08 to 0.16 changes nothing - optimizer already had full freedom at 0.08
- The constraint is "slack" (not binding)

#### Compare to Delay Sensitivity:

| psi_delay | Video_Gen Threshold | Configs Satisfying | Impact on Cost |
|-----------|---------------------|--------------------| ---------------|
| 0.5×      | 3500ms              | 12/60 (20%)        | **HUGE IMPACT** |
| 1.0×      | 7000ms              | 37/60 (62%)        | **Baseline**   |
| 1.5×      | 10500ms             | 52/60 (87%)        | **HUGE IMPACT** |
| 2.0×      | 14000ms             | 58/60 (97%)        | **HUGE IMPACT** |

**Result**: Every change in psi_delay significantly changes feasible set → Affects GPU choices → Changes cost!

---

### 4. Why Delay is More Sensitive to Cost

From cost structure analysis (analyze_cost_structure.py):

```
For equivalent violations:
- 100ms delay violation across all tasks: $562.51/hr
- 2% error rate violation across all tasks: $620.86/hr
```

**But more importantly**:
- **Delay violations are COMMON** because thresholds are tight
- **Error violations are RARE** because thresholds are generous

The total cost is dominated by:
1. **GPU rental costs** - Determined by which GPUs meet delay constraints
2. **Delay penalties** - Many configs violate delay thresholds
3. **Unmet demand penalties** - Rarely triggered (errors are usually within bounds)

---

## Root Cause Summary

### Why Error Constraints Are Not Restrictive:

1. **Generous Thresholds**:
   - Current error thresholds (8-25%) are much higher than typical error rates
   - Example: Code_Gen threshold is 10%, but worst-case error is only 9.47%
   - 83-100% of model-GPU configurations already satisfy thresholds

2. **Weighted Average Formulation**:
   - Error constraint: `Σ e[i,j,k] * x[i,j,k] ≤ ε[i]`
   - Optimizer can blend multiple configs to stay below threshold
   - Flexibility to trade between accuracy and other objectives

3. **No Cost Tradeoff Required**:
   - Can achieve low error rates WITHOUT expensive GPUs
   - Error rates depend more on model size and quantization
   - INT4 quantization increases error by only 35%, and thresholds are still generous

### Why Delay Constraints Are Highly Restrictive:

1. **Tight Thresholds**:
   - Many task-model-GPU combinations exceed delay budgets
   - Example: Video_Gen on 70B model with A6000 = 42 seconds (vs 7 second threshold!)
   - Only 38-100% of configs satisfy thresholds (varies by task)

2. **Absolute Constraint Formulation**:
   - Delay constraint: `Σ d[i,j,k] * (h[i] + f[i]) * x[i,j,k] ≤ Δ[i]`
   - Multiplication by total tokens makes this rigid
   - Cannot easily "blend" configs to meet threshold

3. **Forces Expensive GPU Choices**:
   - To meet tight delay budgets → Must use fast GPUs (H100, A100)
   - Fast GPUs cost 4-6× more than slow GPUs ($2.50/hr vs $0.40/hr)
   - Relaxing delay threshold → Can use cheaper GPUs → Significant cost reduction!

---

## Recommendations

### If You Want Error Constraints to Matter:

#### Option 1: Tighten Error Thresholds (Easiest)
```python
# In parameter_setup.py:159
epsilon = np.array([0.03, 0.04, 0.03, 0.04, 0.06, 0.10])  # Much tighter
```
This would make only 20-40% of configs feasible, forcing quality-cost tradeoffs.

#### Option 2: Increase Base Error Rates
```python
# In parameter_setup.py:389-395
base_errors = np.array([
    np.random.uniform(0.06, 0.10),   # Summarization (was 0.03-0.05)
    np.random.uniform(0.04, 0.06),   # Code_Gen (was 0.02-0.03)
    # ... etc
])
```

#### Option 3: Add Error-Dependent Costs
Add a term to the objective that penalizes expected errors:
```python
# In objective function
C_error = Σ (e_bar[i,j,k] * penalty_per_error * lambda_i * x[i,j,k])
```

### If Current Behavior is Acceptable:

**This is actually realistic!** In practice:
- **Latency SLAs are often the binding constraint** (users notice 100ms delays)
- **Accuracy requirements are often easily met** (most models are "good enough")
- **Cost optimization focuses on meeting latency with minimum GPU cost**

Your model correctly captures that **delay constraints drive GPU selection**, which in turn drives cost. Error rate constraints act as "sanity checks" but don't fundamentally change the optimization.

---

## Validation

To validate this analysis, try this experiment:

```python
# Tighten error thresholds dramatically
data.epsilon = np.array([0.01, 0.01, 0.01, 0.01, 0.02, 0.03])

# Run sensitivity analysis
results_tight = run_sensitivity_analysis(psi_delay=1.0, psi_error=1.0, data)
results_relaxed = run_sensitivity_analysis(psi_delay=1.0, psi_error=2.0, data)

# Now you should see significant cost differences!
```

With tighter thresholds, error sensitivity should emerge because:
- Optimizer forced to choose high-quality models (70B vs 1B)
- May need FP16 instead of INT4 quantization
- These choices affect GPU memory requirements and costs

---

## Conclusion

**Your observation is correct and the model is working as intended.**

Error rate thresholds don't impact cost because they're not binding constraints. The current thresholds are generous enough that the optimizer has many feasible options, so relaxing them further provides no additional flexibility.

Delay constraints, conversely, are tight and force the optimizer to make expensive GPU choices to meet latency requirements. This is why delay sensitivity is high.

**This reflects reality**: In LLM serving, latency SLAs typically drive infrastructure costs more than accuracy requirements.