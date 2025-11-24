import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class LLMInferenceData:
    """Data container for LLM Inference problem parameters"""
    # Problem dimensions
    I: int  # Number of query types
    J: int  # Number of base models
    K: int  # Number of GPU configurations
    N: int  # Number of TP degrees

    # Names
    query_types: List[str]
    model_names: List[str]
    gpu_tiers: List[str]
    tp_degrees: List[int]  # Added: TP degree values {1, 2, 4, 8}

    # Model parameters
    B: np.ndarray  # Model sizes (GB)

    # Query parameters
    h: np.ndarray  # Input token lengths
    f: np.ndarray  # Output token lengths
    lambda_i: np.ndarray  # Query arrival rates

    # GPU parameters
    C_gpu: np.ndarray  # GPU memory capacity
    P_gpu: np.ndarray  # GPU compute power
    p_c: np.ndarray  # GPU rental costs

    # System parameters
    C_storage: float  # Storage capacity
    p_s: float  # Storage cost
    beta: np.ndarray  # KV cache consumption
    alpha: np.ndarray  # Compute consumption
    theta: np.ndarray  # Token sizes

    # Cost parameters
    rho: np.ndarray  # Delay penalties
    phi: np.ndarray  # Unmet demand penalties

    # Thresholds
    delta: float  # Budget threshold
    Delta_T: float  # Rental period
    Delta_i: np.ndarray  # Delay thresholds
    epsilon: np.ndarray  # Error rate thresholds

    # Performance parameters
    d: np.ndarray  # Processing delays
    e: np.ndarray  # Error rates
    T_res: np.ndarray  # Residency times


class DataGenerator:
    """Generate realistic data for LLM Inference optimization"""

    def __init__(self, seed: Optional[int] = 42):
        """Initialize data generator with optional random seed"""
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    def generate(self) -> LLMInferenceData:
        """Generate complete dataset for LLM inference problem"""

        # Problem dimensions
        I, J, K = 6, 6, 12
        N = 4  # Number of TP degrees: {1, 2, 4, 8}
        tp_degrees = [int(1), int(2), int(4), int(8)]  # Ensure Python integers

        # Names
        query_types = ['Summarization', 'Code_Gen', 'Translation',
                      'Math_Solving', 'Image_Gen', 'Video_Gen']

        model_names = [
            'Llama-3.2-1B',
            'Llama-3.2-3B',
            'Llama-3.1-8B',
            'Llama-3.1-70B',
            'Llama-3.1-405B',
            'Llama-3.2-11B-Vision'
        ]

        # GPU tiers ranked from lowest to highest computational power
        gpu_tiers = [
            'A6000_FP16',        # Tier 0: Lowest compute
            'A6000_INT8',        # Tier 1
            'H100_80GB_FP32',    # Tier 2
            'A6000_INT4',        # Tier 3
            'RTX_4090_FP16',     # Tier 4
            'RTX_4090_INT8',     # Tier 5
            'RTX_4090_INT4',     # Tier 6
            'A100_40GB_FP16',    # Tier 7
            'A100_40GB_BF16',    # Tier 8
            'A100_40GB_INT8',    # Tier 9
            'H100_80GB_FP16',    # Tier 10
            'H100_80GB_INT8',    # Tier 11: Highest compute
        ]

        # Fixed parameters
        B = np.array([2, 6, 16, 140, 810, 22])  # Model sizes (GB)
        h = np.array([512, 256, 128, 64, 32, 48])  # Input tokens
        f = np.array([256, 512, 128, 256, 1024, 2048])  # Output tokens

        # GPU specifications (fixed)
        C_gpu = np.array([48, 48, 80, 48, 24, 24, 24, 40, 40, 40, 80, 80])
        P_gpu = np.array([38.7, 58.1, 67, 77.4, 82.6, 123.9, 165.2, 312, 312, 468, 989, 1483.5])

        # Generate random parameters
        lambda_i = self._generate_arrival_rates()
        p_c = self._generate_gpu_costs()
        p_s = np.random.uniform(0.0003, 0.0008)

        beta = np.array([0.02, 0.05, 0.15, 1.4, 8.0, 0.25])
        alpha = self._generate_compute_consumption()
        theta = self._generate_token_sizes()

        rho = self._generate_delay_penalties()
        phi = self._generate_unmet_penalties()

        # Fixed thresholds
        delta = 1000.0
        Delta_T = 1.0
        Delta_i = np.array([1000, 2000, 800, 1500, 3000, 5000])
        epsilon = np.array([0.05, 0.03, 0.05, 0.02, 0.08, 0.10])

        # Generate performance matrices
        d = self._generate_processing_delays(I)  # Returns array of size I
        e = self._generate_error_rates(I)  # Returns array of size I
        T_res = np.ones((I, J, K)) * 0.1
        C_storage = 5000

        return LLMInferenceData(
            I=I, J=J, K=K, N=N,
            query_types=query_types,
            model_names=model_names,
            gpu_tiers=gpu_tiers,
            tp_degrees=tp_degrees,
            B=B, h=h, f=f, lambda_i=lambda_i,
            C_gpu=C_gpu, P_gpu=P_gpu, p_c=p_c,
            C_storage=C_storage, p_s=p_s,
            beta=beta, alpha=alpha, theta=theta,
            rho=rho, phi=phi,
            delta=delta, Delta_T=Delta_T,
            Delta_i=Delta_i, epsilon=epsilon,
            d=d, e=e, T_res=T_res
        )

    def _generate_arrival_rates(self) -> np.ndarray:
        """
        Generate random query arrival rates λ_i (queries per hour).

        Based on typical LLM service workload patterns:
        - Text tasks (Summarization, Translation): High volume
        - Code/Math tasks: Medium volume
        - Media generation: Lower volume due to computational cost

        Reference: "Characterizing LLM Workloads" (Patel et al., 2024)
        """
        return np.array([
            np.random.randint(80, 120),   # Summarization: high demand for document processing
            np.random.randint(60, 100),   # Code_Gen: medium demand from developers
            np.random.randint(100, 140),  # Translation: highest demand (global usage)
            np.random.randint(40, 80),    # Math_Solving: specialized use cases
            np.random.randint(20, 60),    # Image_Gen: lower due to compute cost
            np.random.randint(10, 30)     # Video_Gen: lowest due to extreme cost
        ])

    def _generate_gpu_costs(self) -> np.ndarray:
        """
        Generate random GPU rental costs p_c_k ($/hour) with market variations.

        Base costs from vast.ai marketplace (as of 2024):
        - RTX 4090: $0.40/hr (consumer GPUs, widely available)
        - A6000: $0.65/hr (professional, limited availability)
        - A100: $1.20/hr (data center, on-demand pricing)
        - H100: $2.50/hr (premium, scarce resource)

        Applies ±20% variation to simulate:
        - Spot pricing fluctuations
        - Regional differences
        - Demand-based pricing

        Reference: vast.ai, Lambda Labs, RunPod pricing (2024)
        """
        base_costs = np.array([
            0.65, 0.65, 2.50, 0.65,  # A6000 variants and H100_FP32
            0.40, 0.40, 0.40,         # RTX 4090 variants (best value)
            1.20, 1.20, 1.20,         # A100 variants (enterprise)
            2.50, 2.50                # H100 variants (premium)
        ])

        # Market variation: ±20% from supply/demand dynamics
        return base_costs * np.random.uniform(0.8, 1.2, size=12)

    def _generate_compute_consumption(self) -> np.ndarray:
        """
        Generate compute consumption rates α_i (TFLOPs per 1000 tokens).

        Based on transformer computational complexity O(n²d) where:
        - n = sequence length (tokens)
        - d = model dimension

        Task-specific requirements:
        - Text tasks: Standard transformer ops
        - Code/Math: Additional reasoning layers
        - Media: Cross-attention with visual encoders

        Reference: "Efficient Transformers Survey" (Tay et al., 2022)
        """
        return np.array([
            np.random.uniform(0.3, 0.7),    # Summarization: standard attention
            np.random.uniform(0.6, 1.0),    # Code_Gen: + syntax validation
            np.random.uniform(0.2, 0.4),    # Translation: simple cross-attention
            np.random.uniform(1.0, 1.5),    # Math_Solving: + symbolic reasoning
            np.random.uniform(2.0, 3.0),    # Image_Gen: + diffusion steps
            np.random.uniform(3.5, 4.5)     # Video_Gen: + temporal consistency
        ])

    def _generate_token_sizes(self) -> np.ndarray:
        """
        Generate average token sizes θ_i (MB per token).

        Token size includes:
        - Text tokens: ~4 bytes (UTF-8) + embeddings
        - Image tokens: Latent representations (512-2048 dims)
        - Video tokens: Temporal embeddings + keyframes

        Reference: "Tokenization in Multimodal Models" (Radford et al., 2021)
        """
        return np.array([
            np.random.uniform(0.0008, 0.0012),  # Text: small embeddings
            np.random.uniform(0.0008, 0.0012),  # Code: similar to text
            np.random.uniform(0.0008, 0.0012),  # Translation: text-based
            np.random.uniform(0.0008, 0.0012),  # Math: symbolic representation
            np.random.uniform(0.004, 0.006),    # Image: 5-7x larger (visual tokens)
            np.random.uniform(0.008, 0.012)     # Video: 10-15x larger (temporal)
        ])

    def _generate_delay_penalties(self) -> np.ndarray:
        """
        Generate delay penalty coefficients ρ_i ($ per ms per query).

        Based on user tolerance and business impact:
        - Interactive tasks (Code, Math): High penalty
        - Batch tasks (Summarization): Lower penalty
        - Creative tasks (Media): Users expect delays

        Reference: SLA pricing models from OpenAI, Anthropic (2024)
        """
        return np.array([
            np.random.uniform(0.005, 0.015),  # Summarization: batch processing OK
            np.random.uniform(0.015, 0.025),  # Code_Gen: developers need speed
            np.random.uniform(0.003, 0.008),  # Translation: moderate urgency
            np.random.uniform(0.010, 0.020),  # Math_Solving: interactive tutoring
            np.random.uniform(0.025, 0.035),  # Image_Gen: real-time applications
            np.random.uniform(0.040, 0.060)   # Video_Gen: production pipelines
        ])

    def _generate_unmet_penalties(self) -> np.ndarray:
        """
        Generate unmet demand penalties φ_i ($ per dropped query).

        Based on:
        - Customer lifetime value (CLV) impact
        - Competition risk (user switches to competitor)
        - Task criticality

        Video/Image generation has highest penalty due to:
        - High user acquisition cost
        - Limited alternative providers

        Reference: "Economics of LLM Services" (Chen et al., 2024)
        """
        return np.array([
            np.random.uniform(3, 7),     # Summarization: commoditized service
            np.random.uniform(8, 12),    # Code_Gen: high-value users
            np.random.uniform(2, 4),     # Translation: many alternatives
            np.random.uniform(6, 10),    # Math_Solving: education market
            np.random.uniform(15, 25),   # Image_Gen: creative professionals
            np.random.uniform(25, 35)    # Video_Gen: enterprise customers
        ])

    def _generate_processing_delays(self, I: int) -> np.ndarray:
        """
        Generate processing delay d_i (ms per token) for DETERMINISTIC model.

        For P1 (deterministic), d_i only depends on query type i.
        GPU tier and precision effects are handled in robust optimization via DDU.

        Returns:
            1D array of size I with delay for each query type
        """
        # Generate d_i as a 1D array (only depends on i)
        d = np.array([
            np.random.uniform(0.4, 0.6),    # Summarization: moderate complexity
            np.random.uniform(0.7, 0.9),    # Code_Gen: high complexity
            np.random.uniform(0.3, 0.4),    # Translation: low complexity
            np.random.uniform(1.0, 1.2),    # Math_Solving: reasoning intensive
            np.random.uniform(2.0, 2.5),    # Image_Gen: very high complexity
            np.random.uniform(3.5, 4.0)     # Video_Gen: extreme complexity
        ])

        return d

    def _generate_error_rates(self, I: int) -> np.ndarray:
        """
        Generate error rate e_i (fraction) for DETERMINISTIC model.

        For P1 (deterministic), e_i only depends on query type i.
        GPU tier and precision effects are handled in robust optimization via DDU.

        Returns:
            1D array of size I with error rate for each query type
        """
        # Generate e_i as a 1D array (only depends on i)
        e = np.array([
            np.random.uniform(0.03, 0.05),   # Summarization: moderate error
            np.random.uniform(0.02, 0.03),   # Code_Gen: low error (verifiable)
            np.random.uniform(0.03, 0.04),   # Translation: moderate error
            np.random.uniform(0.05, 0.07),   # Math_Solving: higher error
            np.random.uniform(0.06, 0.08),   # Image_Gen: high error
            np.random.uniform(0.08, 0.10)    # Video_Gen: highest error
        ])

        return e

    def print_summary(self, data: LLMInferenceData):
        """Print summary of generated data"""
        print("\n" + "="*80)
        print("GENERATED DATA SUMMARY")
        print("="*80)

        print(f"\nProblem Size:")
        print(f"  Query Types: {data.I}")
        print(f"  LLM Models: {data.J}")
        print(f"  GPU Configurations: {data.K}")
        print(f"  TP Degrees: {data.N} (values: {data.tp_degrees})")

        print("\nQuery Arrival Rates:")
        for i, qt in enumerate(data.query_types):
            print(f"  {qt:15s}: {data.lambda_i[i]:3d} queries/hr")

        print("\nGPU Costs ($/hr):")
        for k in [0, 6, 11]:  # Show sample: lowest, middle, highest
            print(f"  {data.gpu_tiers[k]:20s}: ${data.p_c[k]:.3f}")

        print(f"\nConstraints:")
        print(f"  Budget: ${data.delta}")
        print(f"  Storage: {data.C_storage} GB")
        print(f"  Delay Thresholds: {data.Delta_i}")
        print(f"  Error Thresholds: {data.epsilon}")


class LLMInferenceOptimizer:
    """Optimization model for LLM Inference Workload Allocation"""

    def __init__(self, data: LLMInferenceData):
        """Initialize optimizer with problem data"""
        self.data = data
        self.model = gp.Model("LLM_Inference_Allocation")
        self.vars = {}  # Store decision variables

    def build_model(self):
        """Build the Gurobi optimization model"""
        self._create_variables()
        self._set_objective()
        self._add_constraints()

    def _create_variables(self):
        """Create all decision variables"""
        d = self.data

        # Workload allocation
        self.vars['x'] = self.model.addVars(d.I, d.J, d.K, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")

        # Number of GPUs
        self.vars['y'] = self.model.addVars(d.J, d.K, vtype=GRB.INTEGER, lb=0, name="y")

        # Placement binary
        self.vars['z'] = self.model.addVars(d.I, d.J, d.K, vtype=GRB.BINARY, name="z")

        # Unmet demand
        self.vars['u'] = self.model.addVars(d.I, d.K, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="u")

        # Tensor parallelism binary variable: w[j,k,n] = 1 if model j on GPU k uses TP degree n
        # Convert to tuple to ensure Gurobi can index properly
        tp_degrees_tuple = tuple(int(n) for n in d.tp_degrees)
        self.vars['w'] = self.model.addVars(d.J, d.K, tp_degrees_tuple, vtype=GRB.BINARY, name="w")

        # Binary indicator: q[j,k] = 1 if model j is deployed on GPU tier k
        self.vars['q'] = self.model.addVars(d.J, d.K, vtype=GRB.BINARY, name="q")

        # Auxiliary variables for bilinear term linearization: x[i,j,k] * w[j,k,n]
        self.vars['xw'] = self.model.addVars(d.I, d.J, d.K, tp_degrees_tuple, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="xw")

        # Auxiliary variables for bilinear term: y[j,k] * w[j,k,n]
        self.vars['yw'] = self.model.addVars(d.J, d.K, tp_degrees_tuple, vtype=GRB.CONTINUOUS, lb=0, name="yw")

    def _set_objective(self):
        """Set the objective function"""
        d = self.data
        x, y, z, u = self.vars['x'], self.vars['y'], self.vars['z'], self.vars['u']

        # C1: Resource rental cost
        C1 = d.Delta_T * gp.quicksum(d.p_c[k] * y[j, k] for j in range(d.J) for k in range(d.K))

        # C2: Storage cost
        C2 = d.Delta_T * gp.quicksum(d.p_s * (d.B[j] * z[i, j, k] + d.theta[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k])
                                     for i in range(d.I) for j in range(d.J) for k in range(d.K))

        # C3: Processing delay penalty
        C3 = gp.quicksum(d.rho[i] * d.d[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k]
                        for i in range(d.I) for j in range(d.J) for k in range(d.K))

        # C4: Unmet demand penalty
        C4 = gp.quicksum(d.phi[i] * u[i, k] * d.lambda_i[i] for i in range(d.I) for k in range(d.K))

        self.model.setObjective(C1 + C2 + C3 + C4, GRB.MINIMIZE)

    def _add_constraints(self):
        """Add all constraints to the model"""
        d = self.data
        x = self.vars['x']
        y = self.vars['y']
        z = self.vars['z']
        u = self.vars['u']
        w = self.vars['w']
        q = self.vars['q']
        xw = self.vars['xw']
        yw = self.vars['yw']

        # Supply-demand balance
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(x[i, j, k] for j in range(d.J) for k in range(d.K)) +
                gp.quicksum(u[i, k] for k in range(d.K)) == 1,
                name=f"supply_demand_{i}")

        # Budget constraint
        self.model.addConstr(
            d.Delta_T * gp.quicksum(d.p_c[k] * y[j, k] + d.p_s * (d.B[j] * z[i, j, k] + d.theta[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k])
                                   for i in range(d.I) for j in range(d.J) for k in range(d.K)) <= d.delta,
            name="budget")

        # GPU memory capacity constraint (using linearized variable yw)
        for j in range(d.J):
            for k in range(d.K):
                self.model.addConstr(
                    d.B[j] * gp.quicksum(yw[j, k, n] for n in d.tp_degrees) +
                    gp.quicksum(d.beta[j] * (d.h[i] + d.f[i]) * d.lambda_i[i] * d.T_res[i, j, k] *
                               gp.quicksum(n * xw[i, j, k, n] for n in d.tp_degrees)
                               for i in range(d.I)) <=
                    d.C_gpu[k] * gp.quicksum(n * yw[j, k, n] for n in d.tp_degrees),
                    name=f"memory_cap_{j}_{k}")

        # Bilinear term linearization: xw[i,j,k,n] = x[i,j,k] * w[j,k,n]
        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    for n in d.tp_degrees:
                        self.model.addConstr(xw[i, j, k, n] <= x[i, j, k], name=f"xw_ub1_{i}_{j}_{k}_{n}")
                        self.model.addConstr(xw[i, j, k, n] <= w[j, k, n], name=f"xw_ub2_{i}_{j}_{k}_{n}")
                        self.model.addConstr(xw[i, j, k, n] >= x[i, j, k] + w[j, k, n] - 1, name=f"xw_lb_{i}_{j}_{k}_{n}")

        # Bilinear term linearization: yw[j,k,n] = y[j,k] * w[j,k,n]
        # Using big-M method where M is the maximum number of GPUs
        M_gpu = 100  # Reasonable upper bound on number of GPUs
        for j in range(d.J):
            for k in range(d.K):
                for n in d.tp_degrees:
                    self.model.addConstr(yw[j, k, n] <= M_gpu * w[j, k, n], name=f"yw_ub1_{j}_{k}_{n}")
                    self.model.addConstr(yw[j, k, n] <= y[j, k], name=f"yw_ub2_{j}_{k}_{n}")
                    self.model.addConstr(yw[j, k, n] >= y[j, k] - M_gpu * (1 - w[j, k, n]), name=f"yw_lb_{j}_{k}_{n}")

        # Tensor parallelism selection: exactly one TP degree per (j,k) if q[j,k] = 1
        for j in range(d.J):
            for k in range(d.K):
                self.model.addConstr(gp.quicksum(w[j, k, n] for n in d.tp_degrees) == q[j, k],
                                    name=f"tp_selection_{j}_{k}")

        # Link q[j,k] with y[j,k]: if y[j,k] > 0, then q[j,k] = 1
        for j in range(d.J):
            for k in range(d.K):
                self.model.addConstr(y[j, k] <= M_gpu * q[j, k], name=f"q_link_ub_{j}_{k}")
                self.model.addConstr(y[j, k] >= q[j, k], name=f"q_link_lb_{j}_{k}")

        # Compute capacity
        for j in range(d.J):
            for k in range(d.K):
                self.model.addConstr(
                    gp.quicksum(d.alpha[i] * (d.f[i] + d.h[i]) * d.lambda_i[i] * x[i, j, k] for i in range(d.I)) <=
                    d.P_gpu[k] * y[j, k],
                    name=f"compute_cap_{j}_{k}")

        # Storage constraint
        self.model.addConstr(
            gp.quicksum(d.B[j] * z[i, j, k] for i in range(d.I) for j in range(d.J) for k in range(d.K)) <= d.C_storage,
            name="storage")

        # Delay constraint
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(d.d[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k] for j in range(d.J) for k in range(d.K)) <= d.Delta_i[i],
                name=f"delay_{i}")

        # Error rate constraint
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(d.e[i] * d.lambda_i[i] * x[i, j, k] for j in range(d.J) for k in range(d.K)) <= d.epsilon[i] * d.lambda_i[i],
                name=f"error_rate_{i}")

        # Placement constraints
        for i in range(d.I):
            for j in range(d.J):
                self.model.addConstr(
                    gp.quicksum(z[i, j, k] for k in range(d.K)) <= 1,
                    name=f"placement1_{i}_{j}"
                )

        for i in range(d.I):
            for k in range(d.K):
                self.model.addConstr(
                    gp.quicksum(z[i, j, k] for j in range(d.J)) <= 1,
                    name=f"placement2_{i}_{k}"
                )

        # Variable bounds
        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    self.model.addConstr(
                        x[i, j, k] <= z[i, j, k],
                        name=f"var_bound_{i}_{j}_{k}"
                    )

    def solve(self, time_limit: int = 300, mip_gap: float = 0.01):
        """Solve the optimization model"""
        self.model.setParam('MIPGap', mip_gap)
        self.model.setParam('TimeLimit', time_limit)
        self.model.optimize()

        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return self._extract_solution()
        else:
            print(f"Optimization failed with status {self.model.status}")
            return None

    def _extract_solution(self) -> Dict:
        """Extract solution from solved model"""
        solution = {
            'objective': self.model.objVal,
            'status': 'OPTIMAL' if self.model.status == GRB.OPTIMAL else 'TIME_LIMIT',
            'gap': self.model.MIPGap if hasattr(self.model, 'MIPGap') else 0,
            'x': {(i, j, k): self.vars['x'][i, j, k].X
                  for i in range(self.data.I)
                  for j in range(self.data.J)
                  for k in range(self.data.K)
                  if self.vars['x'][i, j, k].X > 0.01},
            'y': {(j, k): int(self.vars['y'][j, k].X)
                  for j in range(self.data.J)
                  for k in range(self.data.K)
                  if self.vars['y'][j, k].X > 0.01},
            'w': {(j, k, n): self.vars['w'][j, k, n].X
                  for j in range(self.data.J)
                  for k in range(self.data.K)
                  for n in self.data.tp_degrees
                  if self.vars['w'][j, k, n].X > 0.5},
            'u': {(i, k): self.vars['u'][i, k].X
                  for i in range(self.data.I)
                  for k in range(self.data.K)
                  if self.vars['u'][i, k].X > 0.01}
        }
        return solution

    def display_results(self, solution: Dict):
        """Display optimization results"""
        if solution is None:
            print("No solution to display")
            return

        print("\n" + "="*80)
        print(f"OPTIMIZATION RESULTS ({solution['status']})")
        print("="*80)
        print(f"Total Cost: ${solution['objective']:.2f}")
        if solution['gap'] > 0:
            print(f"Optimality Gap: {solution['gap']:.2%}")
        print("-"*80)

        # GPU allocation with TP degrees
        print("\nGPU ALLOCATION:")
        if solution['y']:
            for (j, k), num_gpus in solution['y'].items():
                if num_gpus > 0:
                    # Find the TP degree for this (j,k) pair
                    tp_degree = None
                    for (jw, kw, n), val in solution['w'].items():
                        if jw == j and kw == k and val > 0.5:
                            tp_degree = n  # n is already the actual TP degree value (1, 2, 4, or 8)
                            break
                    tp_str = f" (TP={tp_degree})" if tp_degree else ""
                    print(f"  {self.data.model_names[j]:20s} on {self.data.gpu_tiers[k]:15s}: {num_gpus} GPUs{tp_str}")

        # Workload distribution
        print("\nWORKLOAD DISTRIBUTION:")
        for i in range(self.data.I):
            allocations = [(j, k, pct) for (qi, j, k), pct in solution['x'].items() if qi == i]
            if allocations:
                print(f"  {self.data.query_types[i]}:")
                for j, k, pct in allocations:
                    queries = int(self.data.lambda_i[i] * pct)
                    print(f"    -> {self.data.model_names[j]} on {self.data.gpu_tiers[k][:10]}: {pct:.1%} ({queries} q/hr)")

        # Unmet demand
        if solution['u']:
            print("\nUNMET DEMAND:")
            for (i, k), unmet in solution['u'].items():
                queries = int(self.data.lambda_i[i] * unmet)
                print(f"  {self.data.query_types[i]} on {self.data.gpu_tiers[k]}: {unmet:.1%} ({queries} queries/hr)")
        else:
            print("\nAll demand satisfied!")


# Main execution
if __name__ == "__main__":
    print("LLM Inference Workload Allocation with Tensor Parallelism")
    print("="*80)

    # Generate data
    print("Generating problem data...")
    generator = DataGenerator(seed=42)
    data = generator.generate()
    generator.print_summary(data)

    # Build and solve model
    print("\nBuilding optimization model...")
    optimizer = LLMInferenceOptimizer(data)
    optimizer.build_model()

    print("Solving optimization model...")
    solution = optimizer.solve(time_limit=300, mip_gap=0.01)

    # Display results
    optimizer.display_results(solution)

    # Test with different random seed
    print("\n" + "="*80)
    print("Testing with different random seed...")
    print("="*80)

    generator2 = DataGenerator(seed=123)
    data2 = generator2.generate()

    optimizer2 = LLMInferenceOptimizer(data2)
    optimizer2.build_model()
    solution2 = optimizer2.solve(time_limit=60, mip_gap=0.02)
    optimizer2.display_results(solution2)