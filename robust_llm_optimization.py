import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class RobustLLMInferenceData:
    """Data container for Robust LLM Inference problem parameters"""
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

    # Robust parameters - Delay uncertainty
    d_bar: np.ndarray  # Nominal processing delays
    d_hat: np.ndarray  # Maximum delay deviations

    # Robust parameters - Error rate uncertainty
    e_bar: np.ndarray  # Nominal error rates
    e_hat: np.ndarray  # Maximum error rate deviations

    T_res: np.ndarray  # Residency times


class RobustDataGenerator:
    """Generate realistic data for Robust LLM Inference optimization"""

    def __init__(self, seed: Optional[int] = 42,
                 delay_uncertainty: float = 0.2,
                 error_uncertainty: float = 0.3):
        """
        Initialize robust data generator

        Args:
            seed: Random seed for reproducibility
            delay_uncertainty: Relative uncertainty level for delay (e.g., 0.2 = ±20%)
            error_uncertainty: Relative uncertainty level for error rate (e.g., 0.3 = ±30%)
        """
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        self.delay_uncertainty = delay_uncertainty
        self.error_uncertainty = error_uncertainty

    def generate(self) -> RobustLLMInferenceData:
        """Generate complete dataset for robust LLM inference problem"""

        # Problem dimensions
        I, J, K = 6, 6, 12
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
            'Llama-3.1-405B',
            'Llama-3.2-11B-Vision'
        ]

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
        delta = 10000
        Delta_T = 12
        Delta_i = np.array([1000, 2000, 800, 1500, 3000, 5000])
        epsilon = np.array([0.05, 0.03, 0.05, 0.02, 0.08, 0.10])

        # Generate nominal values and uncertainty bounds
        d_bar = self._generate_nominal_delays(I)
        d_hat = self._generate_delay_uncertainty(d_bar)

        e_bar = self._generate_nominal_error_rates(I)
        e_hat = self._generate_error_uncertainty(e_bar)

        T_res = np.ones((I, J, K)) * 0.1
        C_storage = 5000

        return RobustLLMInferenceData(
            I=I, J=J, K=K, N=N, TP_degrees=TP_degrees,
            query_types=query_types,
            model_names=model_names,
            gpu_tiers=gpu_tiers,
            B=B, h=h, f=f, lambda_i=lambda_i,
            C_gpu=C_gpu, P_gpu=P_gpu, p_c=p_c,
            C_storage=C_storage, p_s=p_s,
            beta=beta, alpha=alpha, theta=theta,
            rho=rho, phi=phi,
            delta=delta, Delta_T=Delta_T,
            Delta_i=Delta_i, epsilon=epsilon,
            d_bar=d_bar, d_hat=d_hat,
            e_bar=e_bar, e_hat=e_hat,
            T_res=T_res
        )

    def _generate_arrival_rates(self) -> np.ndarray:
        """Generate random query arrival rates λ_i (queries per hour)"""
        return np.array([
            np.random.randint(80, 120),   # Summarization
            np.random.randint(60, 100),   # Code_Gen
            np.random.randint(100, 140),  # Translation
            np.random.randint(40, 80),    # Math_Solving
            np.random.randint(20, 60),    # Image_Gen
            np.random.randint(10, 30)     # Video_Gen
        ])

    def _generate_gpu_costs(self) -> np.ndarray:
        """Generate random GPU rental costs p_c_k ($/hour)"""
        base_costs = np.array([
            0.65, 0.65, 2.50, 0.65,  # A6000 variants and H100_FP32
            0.40, 0.40, 0.40,         # RTX 4090 variants
            1.20, 1.20, 1.20,         # A100 variants
            2.50, 2.50                # H100 variants
        ])
        return base_costs * np.random.uniform(0.8, 1.2, size=12)

    def _generate_compute_consumption(self) -> np.ndarray:
        """Generate compute consumption rates α_i (TFLOPs per 1000 tokens)"""
        return np.array([
            np.random.uniform(0.3, 0.7),    # Summarization
            np.random.uniform(0.6, 1.0),    # Code_Gen
            np.random.uniform(0.2, 0.4),    # Translation
            np.random.uniform(1.0, 1.5),    # Math_Solving
            np.random.uniform(2.0, 3.0),    # Image_Gen
            np.random.uniform(3.5, 4.5)     # Video_Gen
        ])

    def _generate_token_sizes(self) -> np.ndarray:
        """Generate average token sizes θ_i (MB per token)"""
        return np.array([
            np.random.uniform(0.0008, 0.0012),  # Text tasks
            np.random.uniform(0.0008, 0.0012),
            np.random.uniform(0.0008, 0.0012),
            np.random.uniform(0.0008, 0.0012),
            np.random.uniform(0.004, 0.006),    # Image
            np.random.uniform(0.008, 0.012)     # Video
        ])

    def _generate_delay_penalties(self) -> np.ndarray:
        """Generate delay penalty coefficients ρ_i ($ per ms per query)"""
        return np.array([
            np.random.uniform(0.005, 0.015),  # Summarization
            np.random.uniform(0.015, 0.025),  # Code_Gen
            np.random.uniform(0.003, 0.008),  # Translation
            np.random.uniform(0.010, 0.020),  # Math_Solving
            np.random.uniform(0.025, 0.035),  # Image_Gen
            np.random.uniform(0.040, 0.060)   # Video_Gen
        ])

    def _generate_unmet_penalties(self) -> np.ndarray:
        """Generate unmet demand penalties φ_i ($ per dropped query)"""
        return np.array([
            np.random.uniform(3, 7),     # Summarization
            np.random.uniform(8, 12),    # Code_Gen
            np.random.uniform(2, 4),     # Translation
            np.random.uniform(6, 10),    # Math_Solving
            np.random.uniform(15, 25),   # Image_Gen
            np.random.uniform(25, 35)    # Video_Gen
        ])

    def _generate_nominal_delays(self, I: int) -> np.ndarray:
        """
        Generate nominal processing delay d_bar[i] (ms per token)

        These represent expected/average delays under normal conditions
        """
        d_bar = np.array([
            np.random.uniform(0.4, 0.6),    # Summarization
            np.random.uniform(0.7, 0.9),    # Code_Gen
            np.random.uniform(0.3, 0.4),    # Translation
            np.random.uniform(1.0, 1.2),    # Math_Solving
            np.random.uniform(2.0, 2.5),    # Image_Gen
            np.random.uniform(3.5, 4.0)     # Video_Gen
        ])
        return d_bar

    def _generate_delay_uncertainty(self, d_bar: np.ndarray) -> np.ndarray:
        """
        Generate maximum delay deviations d_hat[i]

        d_hat represents the maximum deviation from nominal value
        Actual delay: d[i] ∈ [d_bar[i] - d_hat[i], d_bar[i] + d_hat[i]]
        """
        d_hat = self.delay_uncertainty * d_bar
        return d_hat

    def _generate_nominal_error_rates(self, I: int) -> np.ndarray:
        """
        Generate nominal error rates e_bar[i] (fraction)

        These represent expected error rates under normal conditions
        """
        e_bar = np.array([
            np.random.uniform(0.03, 0.05),   # Summarization
            np.random.uniform(0.02, 0.03),   # Code_Gen
            np.random.uniform(0.03, 0.04),   # Translation
            np.random.uniform(0.05, 0.07),   # Math_Solving
            np.random.uniform(0.06, 0.08),   # Image_Gen
            np.random.uniform(0.08, 0.10)    # Video_Gen
        ])
        return e_bar

    def _generate_error_uncertainty(self, e_bar: np.ndarray) -> np.ndarray:
        """
        Generate maximum error rate deviations e_hat[i]

        e_hat represents the maximum deviation from nominal value
        Actual error rate: e[i] ∈ [e_bar[i] - e_hat[i], e_bar[i] + e_hat[i]]
        """
        e_hat = self.error_uncertainty * e_bar
        # Ensure e_hat doesn't make e_bar - e_hat negative
        e_hat = np.minimum(e_hat, e_bar * 0.9)
        return e_hat

    def print_summary(self, data: RobustLLMInferenceData):
        """Print summary of generated data"""
        print("\n" + "="*80)
        print("ROBUST LLM INFERENCE - DATA SUMMARY")
        print("="*80)

        print(f"\n Problem Size:")
        print(f"  Query Types: {data.I}")
        print(f"  LLM Models: {data.J}")
        print(f"  GPU Configurations: {data.K}")

        print("\n Query Arrival Rates:")
        for i, qt in enumerate(data.query_types):
            print(f"  {qt:15s}: {data.lambda_i[i]:3d} queries/hr")

        print("\n GPU Costs ($/hr):")
        for k in [0, 6, 11]:
            print(f"  {data.gpu_tiers[k]:20s}: ${data.p_c[k]:.3f}")

        print(f"\n Constraints:")
        print(f"  Budget: ${data.delta}")
        print(f"  Storage: {data.C_storage} GB")
        print(f"  Delay Thresholds: {data.Delta_i}")
        print(f"  Error Thresholds: {data.epsilon}")

        print(f"\n Uncertainty Parameters:")
        print(f"  Delay Uncertainty Level: ±{self.delay_uncertainty*100:.0f}%")
        print(f"  Error Uncertainty Level: ±{self.error_uncertainty*100:.0f}%")

        print("\n Nominal Delays (d_bar) and Uncertainty (d_hat):")
        for i, qt in enumerate(data.query_types):
            print(f"  {qt:15s}: d_bar={data.d_bar[i]:.3f} ms/token, " +
                  f"d_hat={data.d_hat[i]:.3f} ms/token " +
                  f"[{data.d_bar[i]-data.d_hat[i]:.3f}, {data.d_bar[i]+data.d_hat[i]:.3f}]")

        print("\n Nominal Error Rates (e_bar) and Uncertainty (e_hat):")
        for i, qt in enumerate(data.query_types):
            print(f"  {qt:15s}: e_bar={data.e_bar[i]:.4f}, " +
                  f"e_hat={data.e_hat[i]:.4f} " +
                  f"[{data.e_bar[i]-data.e_hat[i]:.4f}, {data.e_bar[i]+data.e_hat[i]:.4f}]")


class RobustLLMInferenceOptimizer:
    """Robust Optimization model for LLM Inference Workload Allocation"""

    def __init__(self, data: RobustLLMInferenceData):
        """Initialize robust optimizer with problem data"""
        self.data = data
        self.model = gp.Model("Robust_LLM_Inference")
        self.vars = {}

    def build_model(self):
        """Build the Gurobi robust optimization model"""
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

        # Tensor parallelism selection
        self.vars['w'] = self.model.addVars(d.J, d.K, d.TP_degrees, vtype=GRB.BINARY, name="w")

        self.vars['q'] = self.model.addVars(d.J, d.K, vtype=GRB.BINARY, name="q")

        # Auxiliary variables
        self.vars['v'] = self.model.addVars(d.I, d.J, d.K, d.TP_degrees, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="v")
        self.vars['W'] = self.model.addVars(d.J, d.K, d.TP_degrees, vtype=GRB.BINARY, name="W")

    def _set_objective(self):
        """Set the objective function using NOMINAL delay values"""
        d = self.data
        x, y, z, u = self.vars['x'], self.vars['y'], self.vars['z'], self.vars['u']

        # C1: Resource rental cost
        C1 = d.Delta_T * gp.quicksum(d.p_c[k] * y[j, k] for j in range(d.J) for k in range(d.K))

        # C2: Storage cost
        C2 = d.Delta_T * gp.quicksum(
            d.p_s * (d.B[j] * z[i, j, k] + d.theta[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k])
            for i in range(d.I) for j in range(d.J) for k in range(d.K)
        )

        # C3: Processing delay penalty - using NOMINAL delay (d_bar)
        C3 = gp.quicksum(
            d.rho[i] * d.d_bar[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k]
            for i in range(d.I) for j in range(d.J) for k in range(d.K)
        )

        # C4: Unmet demand penalty
        C4 = gp.quicksum(d.phi[i] * u[i, k] * d.lambda_i[i] for i in range(d.I) for k in range(d.K))

        self.model.setObjective(C1 + C2 + C3 + C4, GRB.MINIMIZE)

    def _add_constraints(self):
        """Add all constraints including robust constraints"""
        d = self.data
        x = self.vars['x']
        y = self.vars['y']
        z = self.vars['z']
        u = self.vars['u']
        w = self.vars['w']
        q = self.vars['q']
        v = self.vars['v']
        W = self.vars['W']

        # Supply-demand balance
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(x[i, j, k] + u[i, k] for j in range(d.J) for k in range(d.K)) == 1,
                name=f"supply_demand_{i}"
            )

        # Budget constraint
        self.model.addConstr(
            d.Delta_T * gp.quicksum(
                d.p_c[k] * y[j, k] +
                d.p_s * (d.B[j] * z[i, j, k] + d.theta[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k])
                for i in range(d.I) for j in range(d.J) for k in range(d.K)
            ) <= d.delta,
            name="budget"
        )

        # TP selection
        for j in range(d.J):
            for k in range(d.K):
                self.model.addConstr(
                    gp.quicksum(w[j, k, n] for n in d.TP_degrees) == q[j, k],
                    name=f"TP_selection_{j}_{k}"
                )

        # Linearization constraints for W = w * q
        for j in range(d.J):
            for k in range(d.K):
                for n in d.TP_degrees:
                    self.model.addConstr(W[j, k, n] <= w[j, k, n])
                    self.model.addConstr(W[j, k, n] <= q[j, k])
                    self.model.addConstr(W[j, k, n] >= w[j, k, n] + q[j, k] - 1)

        # Tensor parallelism relationship
        for j in range(d.J):
            for k in range(d.K):
                self.model.addConstr(
                    y[j, k] == gp.quicksum(n * W[j, k, n] for n in d.TP_degrees),
                    name=f"y_def_{j}_{k}"
                )

        # Routing consistency
        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    self.model.addConstr(
                        z[i, j, k] <= q[j, k],
                        name=f"routing_consistency_{i}_{j}_{k}"
                    )

        # Linearization for v = w * x
        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    for n in d.TP_degrees:
                        self.model.addConstr(v[i, j, k, n] <= w[j, k, n])
                        self.model.addConstr(v[i, j, k, n] <= x[i, j, k])
                        self.model.addConstr(v[i, j, k, n] >= x[i, j, k] - (1 - w[j, k, n]))

        # Memory capacity
        for j in range(d.J):
            for k in range(d.K):
                if d.B[j] > d.C_gpu[k] * max(d.TP_degrees):
                    self.model.addConstr(y[j, k] == 0, name=f"memory_cap_violation_{j}_{k}")
                    continue
                model_weight_term = gp.quicksum((d.B[j] / n) * w[j, k, n] for n in d.TP_degrees)
                kv_cache_term = gp.quicksum(
                    (d.beta[j] / n) * (d.h[i] + d.f[i]) * d.lambda_i[i] * d.T_res[i, j, k] * v[i, j, k, n]
                    for i in range(d.I) for n in d.TP_degrees
                )
                self.model.addConstr(
                    model_weight_term + kv_cache_term <= d.C_gpu[k] * q[j, k],
                    name=f"memory_cap_{j}_{k}"
                )

        # Compute capacity
        for j in range(d.J):
            for k in range(d.K):
                self.model.addConstr(
                    gp.quicksum(d.alpha[i] * (d.f[i] + d.h[i]) * d.lambda_i[i] * x[i, j, k] for i in range(d.I))
                    <= d.P_gpu[k] * y[j, k],
                    name=f"compute_cap_{j}_{k}"
                )

        # Storage constraint
        self.model.addConstr(
            gp.quicksum(d.B[j] * z[i, j, k] for i in range(d.I) for j in range(d.J) for k in range(d.K))
            <= d.C_storage,
            name="storage"
        )

        # ============================================================================
        # ROBUST DELAY CONSTRAINT
        # ============================================================================
        # Worst case: d[i] = d_bar[i] + d_hat[i] (maximum delay)
        # This ensures the delay constraint is satisfied even when delays are at
        # their worst-case values within the uncertainty set
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(
                    (d.d_bar[i] + d.d_hat[i]) * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k]
                    for j in range(d.J) for k in range(d.K)
                ) <= d.Delta_i[i],
                name=f"robust_delay_{i}"
            )

        # ============================================================================
        # ROBUST ERROR RATE CONSTRAINT
        # ============================================================================
        # Worst case: e[i] = e_bar[i] + e_hat[i] (maximum error rate)
        # This ensures the error rate constraint is satisfied even when error rates
        # are at their worst-case values within the uncertainty set
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(
                    (d.e_bar[i] + d.e_hat[i]) * x[i, j, k]
                    for j in range(d.J) for k in range(d.K)
                ) <= d.epsilon[i],
                name=f"robust_error_rate_{i}"
            )

        # Placement constraints
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(z[i, j, k] for j in range(d.J) for k in range(d.K)) == 1,
                name=f"placement_{i}"
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
        """Solve the robust optimization model"""
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
            'u': {(i, k): self.vars['u'][i, k].X
                  for i in range(self.data.I)
                  for k in range(self.data.K)
                  if self.vars['u'][i, k].X > 0.01}
        }
        return solution

    def display_results(self, solution: Dict):
        """Display robust optimization results"""
        if solution is None:
            print("No solution to display")
            return

        print("\n" + "="*80)
        print(f"ROBUST OPTIMIZATION RESULTS ({solution['status']})")
        print("="*80)
        print(f"Total Cost: ${solution['objective']:.2f}")
        if solution['gap'] > 0:
            print(f"Optimality Gap: {solution['gap']:.2%}")
        print("-"*80)

        # GPU allocation
        print("\n GPU ALLOCATION:")
        if solution['y']:
            for (j, k), num_gpus in solution['y'].items():
                if num_gpus > 0:
                    print(f"  {self.data.model_names[j]:20s} on {self.data.gpu_tiers[k]:15s}: {num_gpus} GPUs")
        else:
            print("  No GPUs allocated")

        # Workload distribution
        print("\n WORKLOAD DISTRIBUTION:")
        for i in range(self.data.I):
            allocations = [(j, k, pct) for (qi, j, k), pct in solution['x'].items() if qi == i]
            if allocations:
                print(f"  {self.data.query_types[i]}:")
                for j, k, pct in allocations:
                    queries = int(self.data.lambda_i[i] * pct)
                    print(f"    → {self.data.model_names[j]} on {self.data.gpu_tiers[k][:10]}: {pct:.1%} ({queries} q/hr)")

        # Unmet demand
        if solution['u']:
            print("\n UNMET DEMAND:")
            unmet_by_query = {}
            for (i, k), unmet in solution['u'].items():
                if i not in unmet_by_query:
                    unmet_by_query[i] = 0
                unmet_by_query[i] += unmet
            for i, unmet_total in unmet_by_query.items():
                queries = int(self.data.lambda_i[i] * unmet_total)
                print(f"  {self.data.query_types[i]}: {unmet_total:.1%} ({queries} queries/hr)")
        else:
            print("\n All demand satisfied!")


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("ROBUST LLM INFERENCE WORKLOAD ALLOCATION")
    print("="*80)

    # Generate data with uncertainty
    print("\nGenerating problem data with uncertainty...")
    generator = RobustDataGenerator(seed=42, delay_uncertainty=0.2, error_uncertainty=0.3)
    data = generator.generate()
    generator.print_summary(data)

    # Build and solve robust model
    print("\n" + "="*80)
    print("Building robust optimization model...")
    optimizer = RobustLLMInferenceOptimizer(data)
    optimizer.build_model()

    print("Solving robust optimization model...")
    solution = optimizer.solve(time_limit=300, mip_gap=0.01)

    # Display results
    optimizer.display_results(solution)