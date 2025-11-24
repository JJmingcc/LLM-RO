import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List

class LLMWorkloadAllocation:
    """
    Latency-aware Robust LLM Inference Workload Allocation
    Deterministic formulation using Gurobi optimization
    """

    def __init__(self):
        self.model = None
        self.variables = {}
        self.parameters = {}

    def setup_parameters(self,
                        query_types: List[str],
                        models: List[str],
                        gpu_tiers: List[str],
                        gpu_configs: Dict,
                        query_configs: Dict,
                        model_configs: Dict,
                        system_configs: Dict):
        """
        Setup problem parameters based on paper formulation

        Args:
            query_types: List of query type identifiers (I)
            models: List of base model identifiers (J)
            gpu_tiers: List of GPU tier identifiers (K)
            gpu_configs: GPU configuration parameters
            query_configs: Query type parameters
            model_configs: Model parameters
            system_configs: System-wide parameters
        """

        # Store sets
        self.I = query_types  # Query types
        self.J = models       # Base models
        self.K = gpu_tiers    # GPU tiers

        # Store parameters
        self.gpu_configs = gpu_configs
        self.query_configs = query_configs
        self.model_configs = model_configs
        self.system_configs = system_configs

        # Extract key parameters from configs
        self.p_c = {k: gpu_configs[k]['rental_cost'] for k in self.K}  # Hourly rental rate
        self.p_s = system_configs['storage_rate']  # Storage rate per GB per hour
        self.delta_T = system_configs['rental_period']  # Rental period
        self.delta = system_configs['budget']  # Budget constraint
        self.C_ssd = system_configs['storage_capacity']  # Total SSD capacity

        # Query type parameters
        self.h_i = {i: query_configs[i]['input_tokens'] for i in self.I}
        self.f_i = {i: query_configs[i]['output_tokens'] for i in self.I}
        self.lambda_i = {i: query_configs[i]['arrival_rate'] for i in self.I}
        self.alpha_i = {i: query_configs[i]['compute_per_token'] for i in self.I}
        self.phi_i = {i: query_configs[i]['unmet_penalty'] for i in self.I}
        self.theta_i = {i: query_configs[i]['token_size'] for i in self.I}
        self.Delta_i = {i: query_configs[i]['delay_threshold'] for i in self.I}
        self.epsilon_i = {i: query_configs[i]['error_threshold'] for i in self.I}
        self.rho_i = {i: query_configs[i]['delay_penalty'] for i in self.I}

        # Model parameters
        self.B_j = {j: model_configs[j]['model_size'] for j in self.J}
        self.beta_j = {j: model_configs[j]['kv_cache_per_token'] for j in self.J}

        # GPU tier parameters
        self.C_gpu = {k: gpu_configs[k]['memory_capacity'] for k in self.K}
        self.P_gpu = {k: gpu_configs[k]['compute_power'] for k in self.K}
        self.N_k = {k: gpu_configs[k]['feasible_tp'] for k in self.K}  # Feasible TP degrees

        # Processing delay and error rate (deterministic values for base case)
        self.d_i = {i: query_configs[i]['processing_delay'] for i in self.I}
        self.e_i = {i: query_configs[i]['error_rate'] for i in self.I}

    def create_model(self):
        """Create the Gurobi model with variables and constraints"""

        self.model = gp.Model("LLM_Workload_Allocation")

        # Decision variables
        # x[i,j,k]: workload allocation (number of type-i queries to model j on tier k)
        self.x = self.model.addVars(self.I, self.J, self.K,
                                   vtype=GRB.CONTINUOUS, name="x")

        # y[j,k]: number of tier-k GPUs rented for model j
        self.y = self.model.addVars(self.J, self.K,
                                   vtype=GRB.INTEGER, name="y")

        # z[i,j,k]: binary placement decision (1 if type-i queries assigned to model j at tier k)
        self.z = self.model.addVars(self.I, self.J, self.K,
                                   vtype=GRB.BINARY, name="z")

        # u[i,k]: unmet demand for query type i at tier k
        self.u = self.model.addVars(self.I, self.K,
                                   vtype=GRB.CONTINUOUS, name="u")

        # q[j,k]: binary deployment decision (1 if model j deployed on tier k)
        self.q = self.model.addVars(self.J, self.K,
                                   vtype=GRB.BINARY, name="q")

        # w[n,j,k]: binary TP selection (1 if model j uses TP degree n on tier k)
        self.w = self.model.addVars(
            [(n, j, k) for k in self.K for j in self.J for n in self.N_k[k]],
            vtype=GRB.BINARY, name="w")

        # TP[j,k]: effective tensor parallelism degree
        self.TP = self.model.addVars(self.J, self.K,
                                    vtype=GRB.INTEGER, name="TP")

        # T_res[i,j,k]: residence time for processing
        self.T_res = self.model.addVars(self.I, self.J, self.K,
                                       vtype=GRB.CONTINUOUS, name="T_res")

        # Set objective function
        self.set_objective()

        # Add constraints
        self.add_constraints()

    def set_objective(self):
        """Set the objective function: minimize total cost"""

        # C1: Resource rental cost
        rental_cost = self.delta_T * gp.quicksum(
            self.p_c[k] * self.y[j,k]
            for j in self.J for k in self.K
        )

        # C2: Storage cost
        storage_cost = self.delta_T * gp.quicksum(
            self.p_s * (self.B_j[j] * self.z[i,j,k] +
                       self.theta_i[i] * (self.h_i[i] + self.f_i[i]) *
                       self.lambda_i[i] * self.x[i,j,k])
            for i in self.I for j in self.J for k in self.K
        )

        # C3: Processing delay penalty
        delay_penalty = gp.quicksum(
            self.rho_i[i] * self.d_i[i] * (self.h_i[i] + self.f_i[i]) *
            self.lambda_i[i] * self.x[i,j,k]
            for i in self.I for j in self.J for k in self.K
        )

        # C4: Unmet demand penalty
        unmet_penalty = gp.quicksum(
            self.phi_i[i] * self.lambda_i[i] * self.u[i,k]
            for i in self.I for k in self.K
        )

        # Set objective
        self.model.setObjective(
            rental_cost + storage_cost + delay_penalty + unmet_penalty,
            GRB.MINIMIZE
        )

    def add_constraints(self):
        """Add all constraints to the model"""

        # Constraint (5b): Demand satisfaction
        for i in self.I:
            self.model.addConstr(
                gp.quicksum(self.x[i,j,k] for j in self.J for k in self.K) +
                gp.quicksum(self.u[i,k] for k in self.K) == 1,
                name=f"demand_satisfaction_{i}"
            )

        # Constraint (5c): Budget constraint
        budget_expr = self.delta_T * gp.quicksum(
            self.p_c[k] * self.y[j,k] +
            self.p_s * (self.B_j[j] * self.z[i,j,k] +
                       self.theta_i[i] * (self.h_i[i] + self.f_i[i]) *
                       self.lambda_i[i] * self.x[i,j,k])
            for i in self.I for j in self.J for k in self.K
        )
        self.model.addConstr(budget_expr <= self.delta, name="budget")

        # Constraints (5d): TP selection exclusivity
        for j in self.J:
            for k in self.K:
                self.model.addConstr(
                    gp.quicksum(self.w[n,j,k] for n in self.N_k[k]) == self.q[j,k],
                    name=f"tp_selection_{j}_{k}"
                )

        # Constraints (5e): TP value computation
        for j in self.J:
            for k in self.K:
                self.model.addConstr(
                    self.TP[j,k] == gp.quicksum(n * self.w[n,j,k] for n in self.N_k[k]),
                    name=f"tp_value_{j}_{k}"
                )

        # Constraints (5f): GPU count relationship
        for j in self.J:
            for k in self.K:
                self.model.addConstr(
                    self.y[j,k] == self.TP[j,k] * self.q[j,k],
                    name=f"gpu_count_{j}_{k}"
                )

        # Constraints (5g): Logical consistency
        for i in self.I:
            for j in self.J:
                for k in self.K:
                    self.model.addConstr(
                        self.z[i,j,k] <= self.q[j,k],
                        name=f"logical_consistency_{i}_{j}_{k}"
                    )

        # Constraints (5h): Memory capacity (simplified linearized version)
        for j in self.J:
            for k in self.K:
                # Simplified memory constraint: Model size + KV cache <= GPU memory * num_gpus
                self.model.addConstr(
                    self.B_j[j] * self.q[j,k] +
                    gp.quicksum(
                        self.beta_j[j] * (self.h_i[i] + self.f_i[i]) *
                        self.lambda_i[i] * self.x[i,j,k]
                        for i in self.I
                    ) <= self.C_gpu[k] * self.y[j,k],
                    name=f"memory_capacity_{j}_{k}"
                )

        # Constraints (5i): Compute capacity
        for j in self.J:
            for k in self.K:
                self.model.addConstr(
                    gp.quicksum(
                        self.alpha_i[i] * (self.f_i[i] + self.h_i[i]) *
                        self.lambda_i[i] * self.x[i,j,k]
                        for i in self.I
                    ) <= self.P_gpu[k] * self.y[j,k],
                    name=f"compute_capacity_{j}_{k}"
                )

        # Constraints (5j): Storage capacity
        self.model.addConstr(
            gp.quicksum(
                self.B_j[j] * self.z[i,j,k]
                for i in self.I for j in self.J for k in self.K
            ) <= self.C_ssd,
            name="storage_capacity"
        )

        # Constraints (5k): Delay threshold (deterministic version)
        for i in self.I:
            self.model.addConstr(
                gp.quicksum(
                    self.d_i[i] * (self.h_i[i] + self.f_i[i]) *
                    self.lambda_i[i] * self.x[i,j,k]
                    for j in self.J for k in self.K
                ) <= self.Delta_i[i],
                name=f"delay_threshold_{i}"
            )

        # Constraints (5l): Error rate threshold (deterministic version)
        for i in self.I:
            self.model.addConstr(
                gp.quicksum(
                    self.e_i[i] * self.lambda_i[i] * self.x[i,j,k]
                    for j in self.J for k in self.K
                ) <= self.epsilon_i[i],
                name=f"error_threshold_{i}"
            )

        # Constraints (5m): Single model assignment
        for i in self.I:
            for j in self.J:
                self.model.addConstr(
                    gp.quicksum(self.z[i,j,k] for k in self.K) <= 1,
                    name=f"single_model_{i}_{j}"
                )

        # Constraints (5n): Single tier assignment
        for i in self.I:
            for k in self.K:
                self.model.addConstr(
                    gp.quicksum(self.z[i,j,k] for j in self.J) <= 1,
                    name=f"single_tier_{i}_{k}"
                )

        # Constraints (5o): Workload bounds
        for i in self.I:
            for j in self.J:
                for k in self.K:
                    self.model.addConstr(
                        self.x[i,j,k] <= self.z[i,j,k],
                        name=f"workload_bounds_{i}_{j}_{k}"
                    )
                    self.model.addConstr(
                        self.x[i,j,k] >= 0,
                        name=f"workload_nonneg_{i}_{j}_{k}"
                    )

    def solve(self, time_limit=300, mip_gap=0.01):
        """
        Solve the optimization problem

        Args:
            time_limit: Time limit in seconds
            mip_gap: MIP optimality gap
        """

        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        # Set solver parameters
        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPGap', mip_gap)
        self.model.setParam('OutputFlag', 1)

        # Solve the model
        self.model.optimize()

        # Check solution status
        if self.model.status == GRB.OPTIMAL:
            print(f"Optimal solution found!")
            print(f"Objective value: {self.model.objVal:.4f}")
            return True
        elif self.model.status == GRB.TIME_LIMIT:
            print(f"Time limit reached. Best bound: {self.model.objBound:.4f}")
            print(f"Best solution: {self.model.objVal:.4f}")
            return True
        else:
            print(f"Optimization failed with status: {self.model.status}")
            return False

    def get_solution(self):
        """Extract and return the solution"""

        if self.model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None

        solution = {
            'objective_value': self.model.objVal,
            'workload_allocation': {},
            'gpu_provisioning': {},
            'model_deployment': {},
            'tensor_parallelism': {},
            'unmet_demand': {}
        }

        # Extract workload allocation
        for i in self.I:
            for j in self.J:
                for k in self.K:
                    if self.x[i,j,k].x > 1e-6:
                        solution['workload_allocation'][(i,j,k)] = self.x[i,j,k].x

        # Extract GPU provisioning
        for j in self.J:
            for k in self.K:
                if self.y[j,k].x > 0.5:
                    solution['gpu_provisioning'][(j,k)] = int(self.y[j,k].x)

        # Extract model deployment
        for j in self.J:
            for k in self.K:
                if self.q[j,k].x > 0.5:
                    solution['model_deployment'][(j,k)] = True

        # Extract tensor parallelism
        for j in self.J:
            for k in self.K:
                if self.TP[j,k].x > 0.5:
                    solution['tensor_parallelism'][(j,k)] = int(self.TP[j,k].x)

        # Extract unmet demand
        for i in self.I:
            for k in self.K:
                if self.u[i,k].x > 1e-6:
                    solution['unmet_demand'][(i,k)] = self.u[i,k].x

        return solution

    def print_solution_summary(self):
        """Print a summary of the solution"""

        solution = self.get_solution()
        if solution is None:
            print("No solution available.")
            return

        print("\n" + "="*50)
        print("SOLUTION SUMMARY")
        print("="*50)
        print(f"Total Cost: ${solution['objective_value']:.2f}")

        print("\nWorkload Allocation:")
        for (i,j,k), allocation in solution['workload_allocation'].items():
            print(f"  Query {i} → Model {j} @ Tier {k}: {allocation:.4f}")

        print("\nGPU Provisioning:")
        for (j,k), count in solution['gpu_provisioning'].items():
            print(f"  Model {j} @ Tier {k}: {count} GPUs")

        print("\nTensor Parallelism:")
        for (j,k), tp in solution['tensor_parallelism'].items():
            print(f"  Model {j} @ Tier {k}: TP = {tp}")

        if solution['unmet_demand']:
            print("\nUnmet Demand:")
            for (i,k), unmet in solution['unmet_demand'].items():
                print(f"  Query {i} @ Tier {k}: {unmet:.4f}")