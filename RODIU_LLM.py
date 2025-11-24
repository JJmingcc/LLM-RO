import gurobipy as gp
from gurobipy import GRB, quicksum
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

    # Model parameters
    B: np.ndarray  # Model sizes (GB)
    TP_degrees: np.ndarray  # TP degrees


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
    d: np.ndarray  # Processing delays (I x J x K) - GPU tier dependent
    e: np.ndarray  # Error rates (I x J x K) - GPU precision dependent
    d_bar: np.ndarray  # Nominal processing delays (I x J x K)
    d_hat: np.ndarray  # Processing delay deviations (I x J x K)
    e_bar: np.ndarray  # Nominal error rates (I x J x K)
    e_hat: np.ndarray  # Error rate deviations (I x J x K)
    Gamma_d: int  # Uncertainty budget for delays (scalar)
    Gamma_e: int  # Uncertainty budget for error rates (scalar)
    T_res: np.ndarray  # Residency times
    BigM: int  # BigM value

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
        I, J, K = 6, 6, 10
        N = 4
        TP_degrees = [1, 2, 4, 8]

        # Names
        query_types = ['Summarization', 'Code_Gen', 'Translation',
                      'Math_Solving', 'Image_Gen', 'Video_Gen']

        model_names = [
            'Llama-3.2-1B',
            'Llama-3.2-3B',
            'Llama-3.1-8B',
            'Llama-3.1-70B',
            'Llama-3.2-31B',
            'Llama-3.2-11B-Vision'
        ]

        # GPU tiers ranked from lowest to highest computational power
        gpu_tiers = [
            'A6000_FP16',        # Tier 0: Lowest compute
            'A6000_INT8',        # Tier 1
            'A6000_INT4',        # Tier 2
            'RTX_4090_FP16',     # Tier 3
            'RTX_4090_INT8',     # Tier 4
            'RTX_4090_INT4',     # Tier 5
            'A100_40GB_FP16',    # Tier 6
            'A100_40GB_INT8',    # Tier 7
            'H100_80GB_FP16',    # Tier 8
            'H100_80GB_INT8',    # Tier 9: Highest compute
        ]

        # Fixed parameters
        B = np.array([2, 6, 16, 140, 810, 22])  # Model sizes (GB)
        h = np.array([512, 256, 128, 64, 32, 48])  # Input tokens
        f = np.array([256, 512, 128, 256, 1024, 2048])  # Output tokens

        # GPU specifications (fixed)
        C_gpu = np.array([48, 48, 80, 48, 24, 24, 24, 40, 80, 80])
        P_gpu = np.array([40.7, 58.1, 67, 77.4, 82.6, 123.9, 165.2, 468, 989, 1483.5])

        # Generate random parameters
        lambda_i = self._generate_arrival_rates()
        p_c = self._generate_gpu_costs()
        p_s = np.random.uniform(0.001, 0.005)

        beta = np.array([0.02, 0.05, 0.15, 1.4, 8.0, 0.25])
        alpha = self._generate_compute_consumption()
        theta = self._generate_token_sizes()

        rho = self._generate_delay_penalties()
        phi = self._generate_unmet_penalties()
        # Fixed thresholds
        delta = 3000
        Delta_T = 12
        Delta_i = np.array([1000, 1200, 800, 1500, 4000, 5000])
        epsilon = np.array([0.04, 0.08, 0.03, 0.1, 0.12, 0.15])

        # Generate performance matrices (now considering GPU tier impact)
        d = self._generate_processing_delays(I, J, K, P_gpu)  # Returns array of size I x J x K
        e = self._generate_error_rates(I, J, K, gpu_tiers)  # Returns array of size I x J x K
        d_bar, d_hat = self._generate_uncertainty_delays(d)  # Nominal and deviation
        e_bar, e_hat = self._generate_uncertainty_error_rates(e)  # Nominal and deviation



        # Generate impact factor matrices
        Gamma_d = 50
        Gamma_e = 50

        T_res = np.ones((I, J, K)) * 0.1
        C_storage = 1000
        BigM = 10000
        return LLMInferenceData(
            I=I, J=J, K=K, N=N, TP_degrees=TP_degrees,BigM=BigM,
            query_types=query_types,
            model_names=model_names,
            gpu_tiers=gpu_tiers,
            B=B, h=h, f=f, lambda_i=lambda_i,
            C_gpu=C_gpu, P_gpu=P_gpu, p_c=p_c,
            C_storage=C_storage, p_s=p_s,
            beta=beta, alpha=alpha, theta=theta,
            rho=rho, phi=0.1*phi,
            delta=delta, Delta_T=Delta_T,
            Delta_i=Delta_i, epsilon= epsilon,
            d=d, e=e, d_bar=d_bar, d_hat=d_hat, e_bar=e_bar, e_hat=e_hat,
            Gamma_d=Gamma_d, Gamma_e=Gamma_e, T_res=T_res
        )

    def _generate_arrival_rates(self) -> np.ndarray:
        return np.array([
            np.random.randint(80, 120),   # Summarization
            np.random.randint(60, 100),   # Code_Gen
            np.random.randint(100, 140),  # Translation
            np.random.randint(40, 80),    # Math_Solving
            np.random.randint(20, 60),    # Image_Gen
            np.random.randint(10, 30)     # Video_Gen
        ])

    def _generate_gpu_costs(self) -> np.ndarray:
        base_costs = np.array([
            0.65,  # A6000 variants
            0.65,  # A6000 variants
            2.50,  # H100 variants
            0.65,  # A6000 variants
            0.35,  # RTX 4090 variants
            0.35,  # RTX 4090 variants
            1.20,  # A100 variants
            1.20,  # A100 variants
            2.0,   # H100 variants
            2.50,  # H100 variants
        ])
        return base_costs * np.random.uniform(0.8, 1.2, size=10)

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
        return np.array([
            np.random.uniform(0.0008, 0.0012),
            np.random.uniform(0.0008, 0.0012),
            np.random.uniform(0.0008, 0.0012),
            np.random.uniform(0.0008, 0.0012),
            np.random.uniform(0.004, 0.006),
            np.random.uniform(0.008, 0.012)
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
            np.random.uniform(0.005, 0.010),  # Summarization: batch processing OK
            np.random.uniform(0.015, 0.025),  # Code_Gen: developers need speed
            np.random.uniform(0.005, 0.010),  # Translation: moderate urgency
            np.random.uniform(0.010, 0.020),  # Math_Solving: interactive tutoring
            np.random.uniform(0.020, 0.030),  # Image_Gen: real-time applications
            np.random.uniform(0.030, 0.050)   # Video_Gen: production pipelines
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
            np.random.uniform(30, 50),     # Summarization: commoditized service
            np.random.uniform(80, 100),    # Code_Gen: high-value users
            np.random.uniform(20, 40),     # Translation: many alternatives
            np.random.uniform(80, 100),    # Math_Solving: education market
            np.random.uniform(150, 200),   # Image_Gen: creative professionals
            np.random.uniform(150, 200)    # Video_Gen: enterprise customers
        ])

    def _generate_processing_delays(self, I: int, J: int, K: int, P_gpu: np.ndarray) -> np.ndarray:
        """
        Generate processing delay d[i,j,k] (ms per token) considering GPU tier impact.

        Delays depend on:
        1. Query type (i): Task complexity baseline
        2. Base model (j): Model size affects processing speed
        3. GPU tier (k): Compute power inversely affects delay (higher compute → lower delay)

        Args:
            I: Number of query types
            J: Number of base models
            K: Number of GPU configurations
            P_gpu: GPU compute power array (TFLOPs)

        Returns:
            3D array of size (I, J, K) with delay for each query-model-GPU combination
        """
        # Base delays by query type (task complexity)
        base_delays = np.array([
            np.random.uniform(0.4, 0.6),    # Summarization: moderate complexity
            np.random.uniform(0.7, 0.9),    # Code_Gen: high complexity
            np.random.uniform(0.3, 0.4),    # Translation: low complexity
            np.random.uniform(1.0, 1.2),    # Math_Solving: reasoning intensive
            np.random.uniform(2.0, 2.5),    # Image_Gen: very high complexity
            np.random.uniform(3.5, 4.0)     # Video_Gen: extreme complexity
        ])

        # Model multipliers (larger models take longer per token)
        # Llama-3.2-1B, 3B, 8B, 70B, 31B, 11B-Vision
        model_multipliers = np.array([0.3, 0.5, 1.5, 5, 3.5, 2.5])

        # GPU compute power impact: normalize and invert (higher power → lower delay)
        # Use reference GPU (tier 0) as baseline
        reference_power = P_gpu[0]  # A6000_FP16 baseline
        gpu_speed_factors = reference_power / P_gpu  # Inverse relationship

        # Create 3D tensor: d[i,j,k] = base_delay[i] * model_multiplier[j] * gpu_factor[k]
        d = np.zeros((I, J, K)) 
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    d[i, j, k] = base_delays[i] * model_multipliers[j] * gpu_speed_factors[k]

        return d
    def _generate_error_rates(self, I: int, J: int, K: int, gpu_tiers: List[str]) -> np.ndarray:
        """
        Generate error rate e[i,j,k] (fraction) considering GPU precision impact.

        Error rates depend on:
        1. Query type (i): Task complexity baseline
        2. Base model (j): Model capability (larger models = lower error)
        3. GPU precision (k): Lower precision (INT4, INT8) increases error rates

        Args:
            I: Number of query types
            J: Number of base models
            K: Number of GPU configurations
            gpu_tiers: List of GPU tier names containing precision info (FP16, INT8, INT4)

        Returns:
            3D array of size (I, J, K) with error rate for each query-model-GPU combination
        """
        # Base error rates by query type (task complexity)
        base_errors = np.array([
            np.random.uniform(0.03, 0.05),   # Summarization: moderate error
            np.random.uniform(0.02, 0.03),   # Code_Gen: low error (verifiable)
            np.random.uniform(0.02, 0.03),   # Translation: moderate error
            np.random.uniform(0.05, 0.07),   # Math_Solving: higher error
            np.random.uniform(0.06, 0.08),   # Image_Gen: high error
            np.random.uniform(0.08, 0.10)    # Video_Gen: highest error
        ])

        # Model capacities (higher capacity = more complex model)
        # Llama-3.2-1B, 3B, 8B, 70B, 31B, 11B-Vision
        # Larger models have HIGHER capacity values, which reduces error rates
        model_capacities = np.array([0.5, 1.5, 3, 8, 5, 2.5])

        # GPU precision impact: Extract precision from tier names and create error multipliers
        # FP16: 1.0 (baseline, lowest error)
        # INT8: 1.15 (15% more error due to quantization)
        # INT4: 1.35 (35% more error due to aggressive quantization)
        precision_factors = np.zeros(K)
        for k in range(K):
            tier_name = gpu_tiers[k]
            if 'FP16' in tier_name:
                precision_factors[k] = 1.0  # Baseline precision
            elif 'INT8' in tier_name:
                precision_factors[k] = 1.15  # Moderate quantization error
            elif 'INT4' in tier_name:
                precision_factors[k] = 1.35  # High quantization error
            else:
                precision_factors[k] = 1.0  # Default to FP16

        # Create 3D tensor: e[i,j,k] = (base_error[i] / model_capacity[j]) * precision_factor[k]
        e = np.zeros((I, J, K))
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    e[i, j, k] = (base_errors[i] / model_capacities[j]) * precision_factors[k]

        return e

    def _generate_uncertainty_delays(self, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate nominal delay and maximum deviation for robust optimization.

        The input d represents the nominal/minimum delay (d_bar).
        The actual delay under uncertainty is: d_actual = d_bar + d_hat * ξ, where ξ ∈ [0,1]

        Args:
            d: Nominal delay values (d_bar) from _generate_processing_delays

        Returns:
            Tuple of (d_bar, d_hat) where:
            - d_bar: Nominal/minimum delay (same as input d)
            - d_hat: Maximum deviation (10-25% of d_bar)
        """
        d_bar = d  # d is already the nominal value
        d_hat = d_bar * np.random.uniform(0.10, 0.25, size=d.shape)  # 10-25% of nominal
        return d_bar, d_hat

    def _generate_uncertainty_error_rates(self, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate nominal error rate and maximum deviation for robust optimization.

        The input e represents the nominal/minimum error rate (e_bar).
        The actual error rate under uncertainty is: e_actual = e_bar + e_hat * ξ, where ξ ∈ [0,1]

        Args:
            e: Nominal error rate values (e_bar) from _generate_error_rates

        Returns:
            Tuple of (e_bar, e_hat) where:
            - e_bar: Nominal/minimum error rate (same as input e)
            - e_hat: Maximum deviation (10-20% of e_bar)
        """
        e_bar = e  # e is already the nominal value
        e_hat = e_bar * np.random.uniform(0.10, 0.25, size=e.shape)  # 10-25% of nominal
        return e_bar, e_hat



    def print_summary(self, data: LLMInferenceData):
        """Print summary of generated data"""
        print("\n" + "="*80)
        print("GENERATED DATA SUMMARY")
        print("="*80)

        print(f"\n Problem Size:")
        print(f"  Query Types: {data.I}")
        print(f"  LLM Models: {data.J}")
        print(f"  GPU Configurations: {data.K}")

        print("\n Query Arrival Rates:")
        for i, qt in enumerate(data.query_types):
            print(f"  {qt:15s}: {data.lambda_i[i]:3d} queries/hr")

        print("\n GPU Costs ($/hr):")
        for k in [0, 5, 9]:
            print(f"  {data.gpu_tiers[k]:20s}: ${data.p_c[k]:.3f}")

        print(f"\n Constraints:")
        print(f"  Budget: ${data.delta}")
        print(f"  Storage: {data.C_storage} GB")
        print(f"  Delay Thresholds: {data.Delta_i}")
        print(f"  Error Thresholds: {data.epsilon}")


class LLMInferenceOptimizer:
    """Optimization model for LLM Inference Workload Allocation"""

    def __init__(self, data: LLMInferenceData):
        """Initialize optimizer with problem data"""
        self.data = data
        self.model = None
        self.vars = {}

    def build_and_solve_optimization_problem(self, time_limit: int = 300, mip_gap: float = 0.01):
        """
        Build and solve the complete optimization problem in a single function.

        This unified approach avoids repeated variable definitions and makes the model structure clearer.
        All decision variables, objective function, and constraints are defined together.

        Args:
            time_limit: Maximum solving time in seconds (default: 300)
            mip_gap: MIP gap tolerance for optimization (default: 0.01)

        Returns:
            Solution dictionary containing optimal values, or None if optimization fails
        """
        # Create Gurobi model
        self.model = gp.Model("LLM_Inference_Allocation")
        d = self.data

        # ==================== STEP 1: DEFINE DECISION VARIABLES ====================

        # Primary variables
        x = self.model.addVars(d.I, d.J, d.K, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")  # Workload allocation
        y = self.model.addVars(d.J, d.K, vtype=GRB.INTEGER, lb=0, name="y")  # Number of GPUs
        z = self.model.addVars(d.I, d.J, d.K, vtype=GRB.BINARY, name="z")  # Placement binary
        u = self.model.addVars(d.I, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="u")  # Unmet demand

        # Tensor parallelism variables
        w = self.model.addVars(d.J, d.K, d.TP_degrees, vtype=GRB.BINARY, name="w")  # TP selection
        q = self.model.addVars(d.J, d.K, vtype=GRB.BINARY, name="q")  # GPU configuration active

        # Auxiliary variables for linearization
        v = self.model.addVars(d.I, d.J, d.K, d.TP_degrees, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="v")  # v = w * x
        W = self.model.addVars(d.J, d.K, d.TP_degrees, vtype=GRB.BINARY, name="W")  # W = w * q

        # Robust optimization dual variables
        tau_0 = self.model.addVars(d.I, vtype=GRB.CONTINUOUS, lb=0, name="tau_0")  # Delay constraint duals
        tau_1 = self.model.addVars(d.I, vtype=GRB.CONTINUOUS, lb=0, name="tau_1")  # Error constraint duals
        tau_2 = self.model.addVars(d.I, vtype=GRB.CONTINUOUS, lb=0, name="tau_2")  # Delay objective duals
        sigma_0 = self.model.addVars(d.I, d.J, d.K, vtype=GRB.CONTINUOUS, lb=0, name="sigma_0")
        sigma_1 = self.model.addVars(d.I, d.J, d.K, vtype=GRB.CONTINUOUS, lb=0, name="sigma_1")
        sigma_2 = self.model.addVars(d.I, d.J, d.K, vtype=GRB.CONTINUOUS, lb=0, name="sigma_2")

        varrho = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="varrho")  # Robust delay cost


        # Store variables for later extraction
        self.vars = {
            'x': x, 'y': y, 'z': z, 'u': u, 'w': w, 'q': q, 'v': v, 'W': W,
            'tau_0': tau_0, 'tau_1': tau_1, 'tau_2': tau_2,
            'sigma_0': sigma_0, 'sigma_1': sigma_1, 'sigma_2': sigma_2,'varrho': varrho}

        # ==================== STEP 2: DEFINE OBJECTIVE FUNCTION ====================

        # C1: Resource rental cost (GPU rental over time period)
        C1 = d.Delta_T * gp.quicksum(d.p_c[k] * y[j, k] for j in range(d.J) for k in range(d.K))

        # C2: Storage cost (model storage + KV cache storage)
        C2 = d.Delta_T * gp.quicksum(
            d.p_s * (d.B[j] * z[i, j, k] + d.theta[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k]) for i in range(d.I) for j in range(d.J) for k in range(d.K))

        # C4: Unmet demand penalty
        C4 = gp.quicksum(d.phi[i] * u[i] * d.lambda_i[i] for i in range(d.I))

        # Objective: Minimize total cost (resource + storage + unmet demand + robust delay penalty)
        self.model.setObjective(C1 + C2 + C4 + varrho, GRB.MINIMIZE)


        # === BASIC CONSTRAINTS ===

        # Supply-demand balance: Each query type must be fully allocated or unmet
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(x[i, j, k] for j in range(d.J) for k in range(d.K)) + u[i] == 1, name=f"supply_demand_{i}")

        # unment portion restriction

        for i in range(d.I):
            self.model.addConstr(u[i] <= 0.01, name=f"unmet_portion_restriction")

        # Budget constraint: Total cost cannot exceed budget
        self.model.addConstr(d.Delta_T * gp.quicksum(d.p_c[k] * y[j, k] + d.p_s * (d.B[j] * z[i, j, k] + d.theta[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k]) for i in range(d.I) for j in range(d.J) for k in range(d.K)) <= d.delta, name="budget")

        # === TENSOR PARALLELISM CONSTRAINTS ===
        # TP selection: Exactly one TP degree if GPU config is active
        for j in range(d.J):
            for k in range(d.K):
                self.model.addConstr(
                    gp.quicksum(w[j, k, n] for n in d.TP_degrees) == q[j, k],
                    name=f"TP_selection_{j}_{k}"
                )

        # Linearization for W = w * q (McCormick envelope)
        for j in range(d.J):
            for k in range(d.K):
                for n in d.TP_degrees:
                    self.model.addConstr(W[j, k, n] <= w[j, k, n])
                    self.model.addConstr(W[j, k, n] <= q[j, k])
                    self.model.addConstr(W[j, k, n] >= w[j, k, n] + q[j, k] - 1)

        # Tensor parallelism relationship: y = n * W (number of GPUs = TP degree * active flag)
        for j in range(d.J):
            for k in range(d.K):
                self.model.addConstr(y[j, k] == gp.quicksum(n * W[j, k, n] for n in d.TP_degrees), name=f"y_def_{j}_{k}")

        # === ROUTING AND PLACEMENT CONSTRAINTS ===

        # Routing consistency: Can only route to active GPU configs
        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    self.model.addConstr(z[i, j, k] <= q[j, k], name=f"routing_consistency_{i}_{j}_{k}")

        # Linearization for v = w * x (McCormick envelope)
        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    for n in d.TP_degrees:
                        self.model.addConstr(v[i, j, k, n] <= w[j, k, n])
                        self.model.addConstr(v[i, j, k, n] <= x[i, j, k])
                        self.model.addConstr(v[i, j, k, n] >= x[i, j, k] - (1 - w[j, k, n]))

        # === RESOURCE CAPACITY CONSTRAINTS ===

        # Memory capacity: Model weights + KV cache must fit in GPU memory
        for j in range(d.J):
            for k in range(d.K):
                # Check if model is too large even with max TP degree
                if d.B[j] > d.C_gpu[k] * max(d.TP_degrees):
                    self.model.addConstr(y[j, k] == 0, name=f"memory_cap_violation_{j}_{k}")
                    continue

                model_weight_term = gp.quicksum((d.B[j] / n) * w[j, k, n] for n in d.TP_degrees)
                kv_cache_term = gp.quicksum((d.beta[j] / n) * (d.h[i] + d.f[i]) * d.lambda_i[i] * d.T_res[i, j, k] * v[i, j, k, n]for i in range(d.I) for n in d.TP_degrees)
                self.model.addConstr(
                    model_weight_term + kv_cache_term <= d.C_gpu[k] * q[j, k],
                    name=f"memory_cap_{j}_{k}"
                )

        # Compute capacity: 
        # Total computation must not exceed GPU compute power: the computing power is really tricky to tune -> need to rethink about the computing power for the problem
        for j in range(d.J):
            for k in range(d.K):
                self.model.addConstr(
                    gp.quicksum(d.alpha[i] * (d.f[i] + d.h[i]) * x[i, j, k] for i in range(d.I)) <= d.P_gpu[k] * y[j, k], name=f"compute_cap")


        # Storage constraint: Total model storage must fit in storage capacity
        self.model.addConstr(gp.quicksum(d.B[j] * z[i, j, k] for i in range(d.I) for j in range(d.J) for k in range(d.K)) <= d.C_storage, name="storage")


        # === ROBUST OPTIMIZATION CONSTRAINTS ===

        # Robust delay constraints (with routing-dependent uncertainty)
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(d.d_bar[i, j, k] * (d.h[i] + d.f[i]) * x[i, j, k] for j in range(d.J) for k in range(d.K)) + d.Gamma_d * tau_0[i] +
                gp.quicksum(sigma_0[i, j, k] for j in range(d.J) for k in range(d.K))  <= d.Delta_i[i] * gp.quicksum(n*w[j, k, n] for n in d.TP_degrees), name=f"robust_delay")

        # Robust error rate constraints (with routing-dependent uncertainty)
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(d.e_bar[i, j, k] * x[i, j, k] for j in range(d.J) for k in range(d.K)) + d.Gamma_e * tau_1[i] + gp.quicksum(sigma_1[i, j, k] for j in range(d.J) for k in range(d.K)) <= d.epsilon[i], name=f"robust_error_rate_{i}")

        # Robust delay objective constraint
        self.model.addConstr(varrho >= gp.quicksum(d.Gamma_d * tau_2[i] for i in range(d.I)) + gp.quicksum(sigma_2[i, j, k] for i in range(d.I) for j in range(d.J) for k in range(d.K)), name="robust_delay_obj")

        # === DUAL CONSTRAINTS FOR ROBUST OPTIMIZATION ===

        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    # Dual constraint for delay
                    self.model.addConstr(tau_0[i] + sigma_0[i, j, k] >= d.d_hat[i, j, k] * (d.h[i] + d.f[i]) * x[i, j, k], name=f"dual_delay_{i}_{j}_{k}")
                    # Dual constraint for error rate
                    self.model.addConstr(tau_1[i] + sigma_1[i, j, k] >= d.e_hat[i, j, k] * x[i, j, k], name=f"dual_error_{i}_{j}_{k}")
                    # Dual constraint for delay objective
                    self.model.addConstr(tau_2[i] + sigma_2[i, j, k] >= d.d_hat[i, j, k] * d.rho[i] * d.lambda_i[i] * (d.h[i] + d.f[i]) * x[i, j, k], name=f"dual_delay_obj")

                


        # === LOGICAL CONSTRAINTS ===

        # Placement constraint: Each query type must be placed exactly once
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(z[i, j, k] for j in range(d.J) for k in range(d.K)) == 1,name=f"placement_{i}")

        # Variable bounds: Can only allocate workload if placement is active
        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    self.model.addConstr(x[i, j, k] <= z[i, j, k], name=f"var_bound_{i}_{j}_{k}")

        # ==================== STEP 4: SOLVE OPTIMIZATION PROBLEM ====================

        self.model.setParam('MIPGap', mip_gap)
        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()

        # ==================== STEP 5: EXTRACT AND RETURN SOLUTION ====================

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
            'x': {(i,j,k): self.vars['x'][i,j,k].X
                  for i in range(self.data.I)
                  for j in range(self.data.J)
                  for k in range(self.data.K)
                  if self.vars['x'][i,j,k].X > 0.01},
            'y': {(j,k): int(self.vars['y'][j,k].X)
                  for j in range(self.data.J)
                  for k in range(self.data.K)
                  if self.vars['y'][j,k].X > 0.01},
            'u': {(i): self.vars['u'][i].X
                  for i in range(self.data.I)
                  if self.vars['u'][i].X > 0.01},
            'w': {(j,k,n): self.vars['w'][j,k,n].X
                  for j in range(self.data.J)
                  for k in range(self.data.K)
                  for n in self.data.TP_degrees
                  if self.vars['w'][j,k,n].X > 0.5}
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

        # Display cost components
        self._display_cost_breakdown(solution)

        # Display constraint analysis
        self._display_constraint_analysis(solution)

        # GPU allocation
        print("\n" + "="*80)
        print("GPU ALLOCATION:")
        print("="*80)
        if solution['y']:
            for (j, k), num_gpus in solution['y'].items():
                if num_gpus > 0:
                    print(f"  {self.data.model_names[j]:20s} on {self.data.gpu_tiers[k]:15s}: {num_gpus} GPUs")

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
            unmet_by_query = {}
            for i, unmet in solution['u'].items():
                if i not in unmet_by_query:
                    unmet_by_query[i] = 0
                unmet_by_query[i] += unmet
            for i, unmet_total in unmet_by_query.items():
                queries = int(self.data.lambda_i[i] * unmet_total)
                print(f"  {self.data.query_types[i]}: {unmet_total:.1%} ({queries} queries/hr)")
        else:
            print("\nAll demand satisfied!")

    def _display_cost_breakdown(self, solution: Dict):
        """Display detailed breakdown of all cost components"""
        d = self.data

        print("\n" + "="*80)
        print("COST COMPONENT BREAKDOWN")
        print("="*80)

        # C1: Resource rental cost
        C1 = 0
        for (j, k), y_val in solution['y'].items():
            C1 += d.Delta_T * d.p_c[k] * y_val

        # C2: Storage cost
        C2_model = 0
        C2_data = 0
        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    x_val = solution['x'].get((i, j, k), 0)
                    if x_val > 0:
                        z_val = self.vars['z'][i, j, k].X
                        C2_model += d.Delta_T * d.p_s * d.B[j] * z_val
                        C2_data += d.Delta_T * d.p_s * d.theta[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x_val
        C2 = C2_model + C2_data

        # C4: Unmet demand penalty
        C4 = 0
        for i, u_val in solution['u'].items():
            C4 += d.phi[i] * u_val * d.lambda_i[i]

        # Robust optimization costs
        varrho_val = self.vars['varrho'].X if hasattr(self.vars['varrho'], 'X') else 0

        print(f"\nC1 - Resource Rental Cost:       ${C1:12.2f}  ({C1/solution['objective']*100:5.1f}%)")
        print(f"     GPU rental cost for {d.Delta_T} hours")
        print(f"\nC2 - Storage Cost:                ${C2:12.2f}  ({C2/solution['objective']*100:5.1f}%)")
        print(f"     Model storage:              ${C2_model:12.2f}")
        print(f"     Data storage:               ${C2_data:12.2f}")
        print(f"\nC3 - Robust Delay Penalty:        ${varrho_val:12.2f}  ({varrho_val/solution['objective']*100:5.1f}%)")
        print(f"     Penalty for delay uncertainty")
        print(f"\nC4 - Unmet Demand Penalty:        ${C4:12.2f}  ({C4/solution['objective']*100:5.1f}%)")
        print(f"     Penalty for rejected queries")
        print("-"*80)
        print(f"TOTAL COST:                       ${C1+C2+varrho_val+C4:12.2f}")

    def _display_constraint_analysis(self, solution: Dict):
        """Display LHS vs threshold for all constraints"""
        d = self.data

        print("\n" + "="*80)
        print("CONSTRAINT ANALYSIS (LHS vs Threshold)")
        print("="*80)

        # Budget constraint
        budget_lhs = 0
        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    y_val = solution['y'].get((j, k), 0)
                    x_val = solution['x'].get((i, j, k), 0)
                    z_val = self.vars['z'][i, j, k].X if x_val > 0 else 0
                    budget_lhs += d.Delta_T * (d.p_c[k] * y_val + d.p_s * (d.B[j] * z_val + d.theta[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x_val))
        budget_threshold = d.delta
        budget_util = (budget_lhs / budget_threshold) * 100

        print(f"\n1. BUDGET CONSTRAINT:")
        print(f"   LHS (Actual Cost):        ${budget_lhs:12.2f}")
        print(f"   Threshold (Max Budget):   ${budget_threshold:12.2f}")
        print(f"   Utilization:               {budget_util:6.2f}%")
        print(f"   Status:                    {'SATISFIED' if budget_lhs <= budget_threshold else 'VIOLATED'}")

        # Storage constraint
        storage_lhs = 0
        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    z_val = self.vars['z'][i, j, k].X
                    storage_lhs += d.B[j] * z_val
        storage_threshold = d.C_storage
        storage_util = (storage_lhs / storage_threshold) * 100

        print(f"\n2. STORAGE CONSTRAINT:")
        print(f"   LHS (Total Storage):      {storage_lhs:12.2f} GB")
        print(f"   Threshold (Max Storage):  {storage_threshold:12.2f} GB")
        print(f"   Utilization:               {storage_util:6.2f}%")
        print(f"   Status:                    {'SATISFIED' if storage_lhs <= storage_threshold else 'VIOLATED'}")

        # Computation constraints (per GPU configuration)
        print(f"\n3. COMPUTATION CONSTRAINTS (per Model-GPU pair):")
        print(f"   {'Model':20s} {'GPU':15s} {'LHS (TFLOPs)':>15s} {'Threshold':>15s} {'Util%':>8s} {'Status':>10s}")
        print(f"   {'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*8} {'-'*10}")

        for j in range(d.J):
            for k in range(d.K):
                compute_lhs = 0
                for i in range(d.I):
                    x_val = solution['x'].get((i, j, k), 0)
                    compute_lhs += d.alpha[i] * (d.f[i] + d.h[i]) * x_val
                y_val = solution['y'].get((j, k), 0)
                compute_threshold = d.P_gpu[k] * y_val
                if y_val > 0 or compute_lhs > 0:
                    compute_util = (compute_lhs / compute_threshold * 100) if compute_threshold > 0 else 0
                    status = 'SATISFIED' if compute_lhs <= compute_threshold else 'VIOLATED'
                    print(f"   {d.model_names[j]:20s} {d.gpu_tiers[k]:15s} {compute_lhs:15.2f} {compute_threshold:15.2f} {compute_util:7.1f}% {status:>10s}")

        # Memory constraints (per GPU configuration)
        print(f"\n4. MEMORY CONSTRAINTS (per Model-GPU pair):")
        print(f"   {'Model':20s} {'GPU':15s} {'LHS (GB)':>12s} {'Threshold':>12s} {'Util%':>8s} {'Status':>10s}")
        print(f"   {'-'*20} {'-'*15} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")

        for j in range(d.J):
            for k in range(d.K):
                memory_lhs = 0
                for n in d.TP_degrees:
                    w_val = self.vars['w'][j, k, n].X
                    memory_lhs += (d.B[j] / n) * w_val
                for i in range(d.I):
                    for n in d.TP_degrees:
                        v_val = self.vars['v'][i, j, k, n].X
                        memory_lhs += (d.beta[j] / n) * (d.h[i] + d.f[i]) * d.lambda_i[i] * d.T_res[i, j, k] * v_val
                q_val = self.vars['q'][j, k].X
                memory_threshold = d.C_gpu[k] * q_val
                if q_val > 0 or memory_lhs > 0:
                    memory_util = (memory_lhs / memory_threshold * 100) if memory_threshold > 0 else 0
                    status = 'SATISFIED' if memory_lhs <= memory_threshold else 'VIOLATED'
                    print(f"   {d.model_names[j]:20s} {d.gpu_tiers[k]:15s} {memory_lhs:12.2f} {memory_threshold:12.2f} {memory_util:7.1f}% {status:>10s}")

        # Supply-demand balance constraints (per query type)
        print(f"\n5. SUPPLY-DEMAND BALANCE CONSTRAINTS (per Query Type):")
        print(f"   {'Query Type':20s} {'LHS':>12s} {'Threshold':>12s} {'Status':>10s}")
        print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*10}")

        for i in range(d.I):
            supply_lhs = 0
            for j in range(d.J):
                for k in range(d.K):
                    x_val = solution['x'].get((i, j, k), 0)
                    supply_lhs += x_val
            for k in range(d.K):
                u_val = solution['u'].get((i, k), 0)
                supply_lhs += u_val
            supply_threshold = 1.0
            status = 'SATISFIED' if abs(supply_lhs - supply_threshold) < 0.01 else 'VIOLATED'
            print(f"   {d.query_types[i]:20s} {supply_lhs:12.4f} {supply_threshold:12.4f} {status:>10s}")

        # Unmet demand portion restriction (per query type)
        print(f"\n6. UNMET DEMAND PORTION RESTRICTION (per Query Type):")
        print(f"   {'Query Type':20s} {'LHS':>12s} {'Threshold':>12s} {'Ratio':>8s} {'Status':>10s}")
        print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")

        for i in range(d.I):
            unmet_lhs = sum(solution['u'].get((i, k), 0) for k in range(d.K))
            unmet_threshold = 0.01 * d.lambda_i[i]
            unmet_ratio = unmet_lhs / unmet_threshold if unmet_threshold > 0 else 0
            status = 'SATISFIED' if unmet_lhs <= unmet_threshold else 'VIOLATED'
            print(f"   {d.query_types[i]:20s} {unmet_lhs:12.4f} {unmet_threshold:12.4f} {unmet_ratio:7.2f}x {status:>10s}")

        # Robust delay constraints (per query type)
        print(f"\n7. ROBUST DELAY CONSTRAINTS (per Query Type):")
        print(f"   {'Query Type':20s} {'LHS (ms)':>12s} {'Threshold':>12s} {'Ratio':>8s} {'Status':>10s}")
        print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")

        for i in range(d.I):
            delay_lhs = 0
            for j in range(d.J):
                for k in range(d.K):
                    x_val = solution['x'].get((i, j, k), 0)
                    delay_lhs += d.d_bar[i, j, k] * (d.h[i] + d.f[i]) * x_val
            delay_lhs += d.Gamma_d * self.vars['tau_0'][i].X
            for j in range(d.J):
                for k in range(d.K):
                    delay_lhs += self.vars['sigma_0'][i, j, k].X

            # Calculate TP-adjusted threshold
            delay_threshold_base = d.Delta_i[i]
            tp_multiplier = sum(n * self.vars['w'][j, k, n].X for j in range(d.J) for k in range(d.K) for n in d.TP_degrees)
            delay_threshold = delay_threshold_base * tp_multiplier if tp_multiplier > 0 else delay_threshold_base
            delay_ratio = delay_lhs / delay_threshold if delay_threshold > 0 else 0
            status = 'SATISFIED' if delay_lhs <= delay_threshold else 'VIOLATED'
            print(f"   {d.query_types[i]:20s} {delay_lhs:12.2f} {delay_threshold:12.2f} {delay_ratio:7.2f}x {status:>10s}")

        # Robust error rate constraints (per query type)
        print(f"\n8. ROBUST ERROR RATE CONSTRAINTS (per Query Type):")
        print(f"   {'Query Type':20s} {'LHS':>12s} {'Threshold':>12s} {'Ratio':>8s} {'Status':>10s}")
        print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")

        for i in range(d.I):
            error_lhs = 0
            for j in range(d.J):
                for k in range(d.K):
                    x_val = solution['x'].get((i, j, k), 0)
                    error_lhs += d.e_bar[i, j, k] * x_val
            error_lhs += d.Gamma_e * self.vars['tau_1'][i].X
            for j in range(d.J):
                for k in range(d.K):
                    error_lhs += self.vars['sigma_1'][i, j, k].X

            error_threshold = d.epsilon[i]
            error_ratio = error_lhs / error_threshold if error_threshold > 0 else 0
            status = 'SATISFIED' if error_lhs <= error_threshold else 'VIOLATED'
            print(f"   {d.query_types[i]:20s} {error_lhs:12.4f} {error_threshold:12.4f} {error_ratio:7.2f}x {status:>10s}")


# Main execution
if __name__ == "__main__":
    print("Robust LLM Inference Workload Allocation ")
    print("="*80)

    # Generate data
    print("Generating problem data...")
    generator = DataGenerator(seed=42)
    data = generator.generate()
    generator.print_summary(data)

    # Build and solve model using the unified function
    print("\nBuilding and solving optimization model...")
    optimizer = LLMInferenceOptimizer(data)
    solution = optimizer.build_and_solve_optimization_problem(time_limit=300, mip_gap=0.01)

    # Display results
    optimizer.display_results(solution)
