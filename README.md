# LLM Workload Allocation Optimization

This project implements the deterministic version of the mathematical model from the paper:
**"Latency-aware Robust LLM Inference Workload Allocation under Precision-Dependent Uncertainty"**

## Problem Overview

The optimization model addresses the challenge of efficiently allocating LLM inference workloads across heterogeneous GPU resources while minimizing total cost and satisfying latency, accuracy, and budget constraints.

### Key Decision Variables:
- **x[i,j,k]**: Workload allocation (fraction of type-i queries to model j on GPU tier k)
- **y[j,k]**: Number of tier-k GPUs rented for model j
- **z[i,j,k]**: Binary placement decision (query routing)
- **q[j,k]**: Binary deployment decision (model-GPU assignment)
- **TP[j,k]**: Tensor parallelism degree for model j on tier k

### Objective Function:
Minimize total cost = GPU rental + Storage + Delay penalties + Unmet demand penalties

### Key Constraints:
- Demand satisfaction
- Budget limits
- Memory capacity
- Compute capacity
- Storage limits
- Delay thresholds
- Error rate thresholds
- Logical consistency constraints

## Files

- `llm_workload_allocation.py`: Main optimization model implementation
- `example_usage.py`: Example usage with sample data
- `requirements.txt`: Required Python packages

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from llm_workload_allocation import LLMWorkloadAllocation

# Create and configure the optimizer
optimizer = LLMWorkloadAllocation()
optimizer.setup_parameters(query_types, models, gpu_tiers,
                          gpu_configs, query_configs, model_configs, system_configs)

# Create and solve the model
optimizer.create_model()
success = optimizer.solve(time_limit=300, mip_gap=0.01)

if success:
    optimizer.print_solution_summary()
```

## Sample Configuration

The example includes:
- **Query Types**: Summarization, Code Generation, Translation, Q&A
- **Models**: LLaMA-7B, LLaMA-13B, Mistral-7B
- **GPU Tiers**: T4, A10, A100, H100 with FP16/INT8 precision levels

## Parameter Tuning

The current example shows all demand being dropped as unmet because penalty costs are lower than service costs. To get more realistic results:

### 1. Increase Unmet Demand Penalties
```python
query_configs = {
    'summarization': {
        'unmet_penalty': 500,  # Increase from 45
        # ... other parameters
    }
}
```

### 2. Increase Budget
```python
system_configs = {
    'budget': 5000,  # Increase from 500
    # ... other parameters
}
```

### 3. Adjust Delay/Error Thresholds
Make constraints less restrictive:
```python
query_configs = {
    'summarization': {
        'delay_threshold': 10.0,  # Increase from 5.0
        'error_threshold': 0.1,   # Increase from 0.05
        # ... other parameters
    }
}
```

### 4. Scale Processing Parameters
Reduce processing costs:
```python
query_configs = {
    'summarization': {
        'processing_delay': 0.5,   # Reduce from 2.0
        'delay_penalty': 0.01,     # Reduce from 0.1
        # ... other parameters
    }
}
```

## Model Characteristics

### Current Behavior
The model is finding the mathematically optimal solution: it's cheaper to pay unmet demand penalties than to serve queries given the current parameter settings.

### To Encourage Service
- Increase unmet demand penalties significantly
- Increase budget constraints
- Reduce GPU rental costs
- Adjust delay/error thresholds
- Scale down processing costs

## Extending to Robust Optimization

The current implementation is the deterministic base case. The paper's full robust optimization formulation would add:

1. **Decision-Dependent Uncertainty Sets** for processing delays and error rates
2. **Robust Constraint Reformulation** using duality theory
3. **Bilinear Term Linearization** using McCormick constraints or SOS-1

These extensions handle uncertainty in processing delays and error rates that depend on the chosen GPU configurations and precision levels.

## Results Interpretation

- **Objective Value**: Total system cost ($)
- **Workload Allocation**: Which queries go to which model-GPU combinations
- **GPU Provisioning**: Number of GPUs rented by configuration
- **Tensor Parallelism**: Parallelism degrees for each deployment
- **Unmet Demand**: Fraction of demand not served (penalty applied)

## Academic License

This implementation uses Gurobi with an academic license. For commercial use, a commercial Gurobi license is required.