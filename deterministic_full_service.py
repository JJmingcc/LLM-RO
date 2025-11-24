"""
Deterministic Model with Full Service Requirement (u = 0)

Copy of the deterministic model from Experiment.ipynb, but forcing u = 0
to see if the model is feasible and what happens.
"""

import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


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

        # Generate performance matrices
        d = self._generate_processing_delays(I)
        e = self._generate_error_rates(I)
        T_res = np.ones((I, J, K)) * 0.1
        C_storage = 5000

        return LLMInferenceData(
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
            d=d, e=e, T_res=T_res
        )

    def _generate_arrival_rates(self) -> np.ndarray:
        return np.array([
            np.random.randint(80, 120),
            np.random.randint(60, 100),
            np.random.randint(100, 140),
            np.random.randint(40, 80),
            np.random.randint(20, 60),
            np.random.randint(10, 30)
        ])

    def _generate_gpu_costs(self) -> np.ndarray:
        base_costs = np.array([
            0.65, 0.65, 2.50, 0.65,
            0.40, 0.40, 0.40,
            1.20, 1.20, 1.20,
            2.50, 2.50
        ])
        return base_costs * np.random.uniform(0.8, 1.2, size=12)

    def _generate_compute_consumption(self) -> np.ndarray:
        return np.array([
            np.random.uniform(0.3, 0.7),
            np.random.uniform(0.6, 1.0),
            np.random.uniform(0.2, 0.4),
            np.random.uniform(1.0, 1.5),
            np.random.uniform(2.0, 3.0),
            np.random.uniform(3.5, 4.5)
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
        return np.array([
            np.random.uniform(0.005, 0.015),
            np.random.uniform(0.015, 0.025),
            np.random.uniform(0.003, 0.008),
            np.random.uniform(0.010, 0.020),
            np.random.uniform(0.025, 0.035),
            np.random.uniform(0.040, 0.060)
        ])

    def _generate_unmet_penalties(self) -> np.ndarray:
        return np.array([
            np.random.uniform(3, 7),
            np.random.uniform(8, 12),
            np.random.uniform(2, 4),
            np.random.uniform(6, 10),
            np.random.uniform(15, 25),
            np.random.uniform(25, 35)
        ])

    def _generate_processing_delays(self, I: int) -> np.ndarray:
        d = np.array([
            np.random.uniform(0.4, 0.6),
            np.random.uniform(0.7, 0.9),
            np.random.uniform(0.3, 0.4),
            np.random.uniform(1.0, 1.2),
            np.random.uniform(2.0, 2.5),
            np.random.uniform(3.5, 4.0)
        ])
        return d

    def _generate_error_rates(self, I: int) -> np.ndarray:
        e = np.array([
            np.random.uniform(0.03, 0.05),
            np.random.uniform(0.02, 0.03),
            np.random.uniform(0.03, 0.04),
            np.random.uniform(0.05, 0.07),
            np.random.uniform(0.06, 0.08),
            np.random.uniform(0.08, 0.10)
        ])
        return e


class LLMInferenceOptimizerFullService:
    """Optimization model with FULL SERVICE requirement (u = 0)"""

    def __init__(self, data: LLMInferenceData):
        self.data = data
        self.model = gp.Model("LLM_Inference_Full_Service")
        self.vars = {}

    def build_model(self):
        self._create_variables()
        self._set_objective()
        self._add_constraints()

    def _create_variables(self):
        d = self.data

        # Workload allocation
        self.vars['x'] = self.model.addVars(d.I, d.J, d.K, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")

        # Number of GPUs
        self.vars['y'] = self.model.addVars(d.J, d.K, vtype=GRB.INTEGER, lb=0, name="y")

        # Placement binary
        self.vars['z'] = self.model.addVars(d.I, d.J, d.K, vtype=GRB.BINARY, name="z")

        # NO unmet demand variable!

        # Tensor parallelism selection
        self.vars['w'] = self.model.addVars(d.J, d.K, d.TP_degrees, vtype=GRB.BINARY, name="w")

        self.vars['q'] = self.model.addVars(d.J, d.K, vtype=GRB.BINARY, name="q")

        # Auxiliary variables
        self.vars['v'] = self.model.addVars(d.I, d.J, d.K, d.TP_degrees, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="v")
        self.vars['W'] = self.model.addVars(d.J, d.K, d.TP_degrees, vtype=GRB.BINARY, name="W")

    def _set_objective(self):
        d = self.data
        x, y, z = self.vars['x'], self.vars['y'], self.vars['z']

        # C1: Resource rental cost
        C1 = d.Delta_T * gp.quicksum(d.p_c[k] * y[j, k] for j in range(d.J) for k in range(d.K))

        # C2: Storage cost
        C2 = d.Delta_T * gp.quicksum(d.p_s * (d.B[j] * z[i, j, k] + d.theta[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k]) for i in range(d.I) for j in range(d.J) for k in range(d.K))

        # C3: Processing delay penalty
        C3 = gp.quicksum(d.rho[i] * d.d[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k] for i in range(d.I) for j in range(d.J) for k in range(d.K))

        # NO C4: No unmet demand

        self.model.setObjective(C1 + C2 + C3, GRB.MINIMIZE)

    def _add_constraints(self):
        d = self.data
        x = self.vars['x']
        y = self.vars['y']
        z = self.vars['z']
        w = self.vars['w']
        q = self.vars['q']
        v = self.vars['v']
        W = self.vars['W']

        # FULL SERVICE: All demand must be served
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(x[i, j, k] for j in range(d.J) for k in range(d.K)) == 1,
                name=f"full_service_{i}"
            )

        # Budget constraint
        self.model.addConstr(
            d.Delta_T * gp.quicksum(d.p_c[k] * y[j, k] + d.p_s * (d.B[j] * z[i, j, k] + d.theta[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k]) for i in range(d.I) for j in range(d.J) for k in range(d.K)) <= d.delta,
            name="budget"
        )

        # TP selection
        for j in range(d.J):
            for k in range(d.K):
                self.model.addConstr(gp.quicksum(w[j,k,n] for n in d.TP_degrees) == q[j,k], name=f"TP_selection_{j}_{k}")

        # Linearization constraints for W = w * q
        for j in range(d.J):
            for k in range(d.K):
                for n in d.TP_degrees:
                    self.model.addConstr(W[j,k,n] <= w[j,k,n])
                    self.model.addConstr(W[j,k,n] <= q[j,k])
                    self.model.addConstr(W[j,k,n] >= w[j,k,n] + q[j,k] - 1)

        # Tensor parallelism relationship
        for j in range(d.J):
            for k in range(d.K):
                self.model.addConstr(y[j, k] == gp.quicksum(n * W[j, k, n] for n in d.TP_degrees), name=f"y_def_{j}_{k}")

        # Routing consistency constraint
        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    self.model.addConstr(z[i,j,k] <= q[j,k], name=f"routing_consistency_{i}_{j}_{k}")

        # Linearization for v = w * x
        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    for n in d.TP_degrees:
                        self.model.addConstr(v[i,j,k,n] <= w[j,k,n])
                        self.model.addConstr(v[i,j,k,n] <= x[i,j,k])
                        self.model.addConstr(v[i,j,k,n] >= x[i,j,k] - (1 - w[j,k,n]))

        # Memory capacity
        for j in range(d.J):
            for k in range(d.K):
                if d.B[j] > d.C_gpu[k] * max(d.TP_degrees):
                    self.model.addConstr(y[j,k] == 0, name=f"memory_cap_violation_{j}_{k}")
                    continue
                model_weight_term = gp.quicksum((d.B[j] / n) * w[j, k, n] for n in d.TP_degrees)
                kv_cache_term = gp.quicksum((d.beta[j] / n) * (d.h[i] + d.f[i]) * d.lambda_i[i] * d.T_res[i, j, k] * v[i, j, k, n] for i in range(d.I) for n in d.TP_degrees)
                self.model.addConstr(model_weight_term + kv_cache_term <= d.C_gpu[k] * q[j, k], name=f"memory_cap_{j}_{k}")

        # Compute capacity
        for j in range(d.J):
            for k in range(d.K):
                self.model.addConstr(gp.quicksum(d.alpha[i] * (d.f[i] + d.h[i]) * d.lambda_i[i] * x[i, j, k] for i in range(d.I)) <= d.P_gpu[k] * y[j, k], name=f"compute_cap_{j}_{k}")

        # Storage constraint
        self.model.addConstr(gp.quicksum(d.B[j] * z[i, j, k] for i in range(d.I) for j in range(d.J) for k in range(d.K)) <= d.C_storage, name="storage")

        # Delay constraint (ORIGINAL - keeping the bug to see what happens)
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(d.d[i] * (d.h[i] + d.f[i]) * d.lambda_i[i] * x[i, j, k] for j in range(d.J) for k in range(d.K)) <= d.Delta_i[i],
                name=f"delay_{i}"
            )

        # Error rate constraint
        for i in range(d.I):
            self.model.addConstr(
                gp.quicksum(d.e[i] * x[i, j, k] for j in range(d.J) for k in range(d.K)) <= d.epsilon[i],
                name=f"error_rate_{i}"
            )

        # Placement constraints
        for i in range(d.I):
            self.model.addConstr(gp.quicksum(z[i, j, k] for j in range(d.J) for k in range(d.K)) == 1, name=f"placement_{i}")

        # Variable bounds
        for i in range(d.I):
            for j in range(d.J):
                for k in range(d.K):
                    self.model.addConstr(x[i, j, k] <= z[i, j, k], name=f"var_bound_{i}_{j}_{k}")

    def solve(self, time_limit: int = 300, mip_gap: float = 0.01):
        self.model.setParam('MIPGap', mip_gap)
        self.model.setParam('TimeLimit', time_limit)
        self.model.optimize()

        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return self._extract_solution()
        elif self.model.status == GRB.INFEASIBLE:
            print("\n" + "="*80)
            print("MODEL IS INFEASIBLE!")
            print("="*80)
            print("Computing IIS...")
            self.model.computeIIS()
            print("\nConflicting constraints:")
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print(f"  - {c.ConstrName}")
            return None
        else:
            print(f"Optimization failed with status {self.model.status}")
            return None

    def _extract_solution(self) -> Dict:
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
            'u': {}
        }
        return solution

    def display_results(self, solution: Dict):
        if solution is None:
            print("No solution to display")
            return

        print("\n" + "="*80)
        print(f"FULL SERVICE RESULTS ({solution['status']})")
        print("="*80)
        print(f"Total Cost: ${solution['objective']:.2f}")
        if solution['gap'] > 0:
            print(f"Optimality Gap: {solution['gap']:.2%}")

        # GPU allocation
        print("\n GPU ALLOCATION:")
        if solution['y']:
            total_gpus = sum(solution['y'].values())
            print(f"  Total GPUs: {total_gpus}")
            for (j, k), num_gpus in solution['y'].items():
                if num_gpus > 0:
                    print(f"  {self.data.model_names[j]:20s} on {self.data.gpu_tiers[k]:15s}: {num_gpus} GPUs")

        # Workload distribution
        print("\n WORKLOAD DISTRIBUTION:")
        for i in range(self.data.I):
            allocations = [(j, k, pct) for (qi, j, k), pct in solution['x'].items() if qi == i]
            if allocations:
                print(f"  {self.data.query_types[i]} ({self.data.lambda_i[i]} queries/hr):")
                for j, k, pct in allocations:
                    queries = int(self.data.lambda_i[i] * pct)
                    print(f"    → {self.data.model_names[j]} on {self.data.gpu_tiers[k][:10]}: {pct:.1%} ({queries} q/hr)")

        print("\n ✅ ALL DEMAND SATISFIED (100% service rate)")


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("DETERMINISTIC MODEL WITH FULL SERVICE REQUIREMENT (u = 0)")
    print("="*80)

    generator = DataGenerator(seed=42)
    data = generator.generate()

    print(f"\nParameters:")
    print(f"  Budget: ${data.delta:,}")
    print(f"  Total Demand: {sum(data.lambda_i)} queries/hr")
    print(f"  Forcing: 100% service rate (u = 0)")

    optimizer = LLMInferenceOptimizerFullService(data)
    optimizer.build_model()
    solution = optimizer.solve(time_limit=300, mip_gap=0.01)

    optimizer.display_results(solution)