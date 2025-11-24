"""
System Parameters Setup for LLM Inference Optimization

This module contains all parameter definitions and data generation logic
used by both deterministic (LLM_DET) and robust (RDDU) optimization models.

Includes:
- Common system parameters (shared across all models)
- Uncertainty parameters (for robust optimization models)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


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
    d_bar: np.ndarray  # Nominal processing delays
    e_bar: np.ndarray  # Nominal error rates
    T_res: np.ndarray  # Residency times


@dataclass
class RobustLLMInferenceData(LLMInferenceData):
    """Extended data container for robust optimization with uncertainty parameters"""
    # Uncertainty parameters for robust optimization
    d_bar: np.ndarray  # Nominal processing delays (I x J x K)
    d_hat: np.ndarray  # Processing delay deviations (I x J x K)
    e_bar: np.ndarray  # Nominal error rates (I x J x K)
    e_hat: np.ndarray  # Error rate deviations (I x J x K)
    Gamma_d: int  # Uncertainty budget for delays (scalar)
    Gamma_e: int  # Uncertainty budget for error rates (scalar)
    BigM: int  # BigM value for linearization


class ParameterGenerator:
    """
    Generate system parameters for LLM Inference optimization.

    This class provides methods to generate both common parameters (used by all models)
    and uncertainty-specific parameters (used by robust optimization models).
    """

    def __init__(self, seed: Optional[int] = 42):
        """Initialize parameter generator with optional random seed"""
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    # ==================== FIXED PARAMETERS ====================

    def get_problem_dimensions(self):
        """Return standard problem dimensions"""
        I, J, K = 6, 6, 10
        N = 4
        TP_degrees = [1, 2, 4, 8]
        return I, J, K, N, TP_degrees

    def get_query_types(self):
        """Return list of query type names"""
        return ['Summarization', 'Code_Gen', 'Translation',
                'Math_Solving', 'Image_Gen', 'Video_Gen']

    def get_model_names(self):
        """Return list of model names"""
        return [
            'Llama-3.2-1B',
            'Llama-3.2-3B',
            'Llama-3.1-8B',
            'Llama-3.1-70B',
            'Llama-3.2-31B',
            'Llama-3.2-11B-Vision'
        ]

    def get_gpu_tiers(self):
        """Return list of GPU tier names ranked from lowest to highest computational power"""
        return [
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

    def get_model_sizes(self):
        """Return model sizes in GB"""
        return np.array([2, 6, 16, 140, 810, 22])  # Model sizes (GB)

    def get_input_token_lengths(self):
        """Return input token lengths for each query type"""
        return np.array([512, 256, 128, 64, 32, 48])  # Input tokens

    def get_output_token_lengths(self):
        """Return output token lengths for each query type"""
        return np.array([256, 512, 128, 256, 1024, 2048])  # Output tokens

    def get_gpu_memory_capacities(self):
        """Return GPU memory capacities in GB"""
        return np.array([48, 48, 80, 48, 24, 24, 24, 40, 80, 80])

    def get_gpu_compute_powers(self):
        """Return GPU compute powers in TFLOPs"""
        return np.array([40.7, 58.1, 67, 77.4, 82.6, 123.9, 165.2, 468, 989, 1483.5])

    def get_kv_cache_consumption(self):
        """Return KV cache consumption rates per model"""
        return np.array([0.02, 0.05, 0.15, 1.4, 8.0, 0.25])

    def get_thresholds(self):
        """Return system thresholds"""
        delta = 5000  # Budget threshold
        Delta_T = 12  # Rental period
        Delta_i = np.array([1000, 1500, 800, 2000, 4000, 5000])  # Delay thresholds
        epsilon = np.array([0.08, 0.1, 0.08, 0.1, 0.15, 0.25])  # Error rate thresholds
        C_storage = 1000  # Storage capacity
        return delta, Delta_T, Delta_i, epsilon, C_storage

    def get_bigm_value(self):
        """Return BigM value for linearization in robust optimization"""
        return 10000

    # ==================== RANDOM PARAMETER GENERATORS ====================

    def generate_arrival_rates(self) -> np.ndarray:
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

    def generate_gpu_costs(self) -> np.ndarray:
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
        # Market variation: ±20% from supply/demand dynamics
        return base_costs * np.random.uniform(0.8, 1.2, size=10)

    def generate_storage_cost(self) -> float:
        """Generate random storage cost per GB per hour"""
        return np.random.uniform(0.001, 0.005)

    def generate_compute_consumption(self) -> np.ndarray:
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

    def generate_token_sizes(self) -> np.ndarray:
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

    def generate_delay_penalties(self) -> np.ndarray:
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
            np.random.uniform(0.025, 0.035),  # Image_Gen: real-time applications
            np.random.uniform(0.040, 0.060)   # Video_Gen: production pipelines
        ])

    def generate_unmet_penalties(self) -> np.ndarray:
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

    def generate_processing_delays(self, I: int = 6, J: int = 6, K: int = 10) -> np.ndarray:
        """
        Generate nominal processing delay d_bar[i,j,k] (ms per token) as a 3D array.

        Delays depend on:
        1. Query type (i): Task complexity baseline
        2. Base model (j): Model size affects processing speed
        3. GPU tier (k): GPU compute power (higher power → lower delay)

        Formula: d[i,j,k] = base_delay[i] * model_multiplier[j] * gpu_speed_factor[k]

        Args:
            I: Number of query types (default 6)
            J: Number of base models (default 6)
            K: Number of GPU configurations (default 10)

        Returns:
            3D array of size (I, J, K) with nominal delay for each query-model-GPU combination
        """
        # Base delays by query type (task complexity in ms/token)
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

        # GPU speed factors based on compute power
        # Higher GPU power → lower delay (inverse relationship)
        P_gpu = self.get_gpu_compute_powers()  # [40.7, 58.1, 67, 77.4, 82.6, 123.9, 165.2, 468, 989, 1483.5]
        reference_power = P_gpu[0]  # A6000_FP16 as baseline
        gpu_speed_factors = reference_power / P_gpu  # Inverse: higher power = lower factor = lower delay

        # Create 3D tensor: d[i,j,k] = base_delay[i] * model_multiplier[j] * gpu_speed_factor[k]
        d = np.zeros((I, J, K))
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    d[i, j, k] = base_delays[i] * model_multipliers[j] * gpu_speed_factors[k]

        return d

    def generate_error_rates(self, I: int = 6, J: int = 6, K: int = 10) -> np.ndarray:
        """
        Generate nominal error rate e_bar[i,j,k] (fraction) as a 3D array.

        Error rates depend on:
        1. Query type (i): Task complexity baseline
        2. Base model (j): Model capability (larger models = lower error)
        3. GPU precision (k): Quantization level affects accuracy
           - FP16: Baseline precision (factor = 1.0)
           - INT8: Moderate quantization error (factor = 1.15)
           - INT4: High quantization error (factor = 1.35)

        Formula: e[i,j,k] = (base_error[i] / model_capacity[j]) * precision_factor[k]

        Args:
            I: Number of query types (default 6)
            J: Number of base models (default 6)
            K: Number of GPU configurations (default 10)

        Returns:
            3D array of size (I, J, K) with nominal error rate for each query-model-GPU combination
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

        # GPU precision impact: Extract precision from tier names
        # FP16: 1.0 (baseline, lowest error)
        # INT8: 1.15 (15% more error due to quantization)
        # INT4: 1.35 (35% more error due to aggressive quantization)
        gpu_tiers = self.get_gpu_tiers()
        precision_factors = np.zeros(K)
        for k in range(K):
            tier_name = gpu_tiers[k]
            if 'FP16' in tier_name or 'BF16' in tier_name:
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

    def generate_residency_times(self, I: int, J: int, K: int) -> np.ndarray:
        """Generate residency times for KV cache"""
        return np.ones((I, J, K)) * 0.1

    # ==================== UNCERTAINTY PARAMETERS (FOR ROBUST OPTIMIZATION) ====================

    def generate_uncertainty_delays(self, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate nominal and deviation values for processing delays used in robust optimization.

        Takes the nominal delay values and generates uncertainty parameters:
        - d_bar: Nominal delay (baseline performance)
        - d_hat: Maximum deviation from nominal (10-25% of d_bar)

        Args:
            d: Nominal delay values from generate_processing_delays (I x J x K array)

        Returns:
            Tuple of (d_bar, d_hat) where:
            - d_bar: Nominal delay (I x J x K array, same as input d)
            - d_hat: Maximum deviation from nominal (I x J x K array, 10-25% of d_bar)
        """
        d_bar = d  # d is already the nominal value
        d_hat = d_bar * np.random.uniform(0.10, 0.25, size=d.shape)  # 10-25% of nominal
        return d_bar, d_hat

    def generate_uncertainty_error_rates(self, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate nominal and deviation values for error rates used in robust optimization.

        Takes the nominal error rate values and generates uncertainty parameters:
        - e_bar: Nominal error rate (baseline performance)
        - e_hat: Maximum deviation from nominal (10-25% of e_bar)

        Args:
            e: Nominal error rate values from generate_error_rates (I x J x K array)

        Returns:
            Tuple of (e_bar, e_hat) where:
            - e_bar: Nominal error rate (I x J x K array, same as input e)
            - e_hat: Maximum deviation from nominal (I x J x K array, 10-25% of e_bar)
        """
        e_bar = e  # e is already the nominal value
        e_hat = e_bar * np.random.uniform(0.10, 0.25, size=e.shape)  # 10-25% of nominal
        return e_bar, e_hat

    def generate_uncertainty_budgets(self, I: int, J: int) -> Tuple[int, int]:
        """
        Generate uncertainty budgets for delay and error rate constraints.

        The uncertainty budget Γ controls how many uncertain parameters can
        simultaneously deviate from their nominal values. This implements
        budgeted uncertainty sets for computational tractability.

        Args:
            I: Number of query types
            J: Number of base models

        Returns:
            Tuple of (Gamma_d, Gamma_e) - uncertainty budgets for delays and error rates
        """
        # Fixed uncertainty budgets for sensitivity analysis
        Gamma_d = 10
        Gamma_e = 10
        return Gamma_d, Gamma_e

    def generate_gamma_impact(self, J: int, K: int, model_names: List[str],P_gpu: np.ndarray) -> np.ndarray:
        """
        Generate delay uncertainty impact factors γ[j,k] for routing-dependent uncertainty.

        Impact factor represents how GPU tier affects delay uncertainty:
        - Higher values: More uncertainty when model-GPU pairing is suboptimal
        - Lower values: Less uncertainty when model-GPU pairing is optimal

        Depends on:
        - Model complexity: More complex models benefit more from powerful GPUs
        - GPU compute power: More powerful GPUs reduce uncertainty for complex models

        Args:
            J: Number of base models
            K: Number of GPU configurations
            model_names: List of model names
            P_gpu: GPU compute powers

        Returns:
            2D array (J x K) with impact factors in [0, 1]
        """
        model_complexity = {
            'Llama-3.2-1B': 'low',
            'Llama-3.2-3B': 'low',
            'Llama-3.1-8B': 'medium',
            'Llama-3.2-11B-Vision': 'medium',
            'Llama-3.1-70B': 'high',
            'Llama-3.2-31B': 'high'
        }

        def classify_gpu_power(compute_power):
            if compute_power < 150:
                return 'low'
            elif compute_power < 400:
                return 'medium'
            elif compute_power < 900:
                return 'high'
            else:
                return 'luxury'

        impact_ranges = {
            'low': (0.2, 0.25),
            'medium': (0.40, 0.45),
            'high': (0.60, 0.65),
            'luxury': (0.80, 0.85)
        }

        gamma_impact = np.zeros((J, K))

        for j, model in enumerate(model_names):
            for k in range(K):
                m_complexity = model_complexity[model]
                gpu_power = classify_gpu_power(P_gpu[k])

                complexity_level = ['low', 'medium', 'high', 'luxury'].index(m_complexity)
                power_level = ['low', 'medium', 'high', 'luxury'].index(gpu_power)

                # REVISED LOGIC: Higher GPU power = Higher impact on reducing delay
                # impact_score = power_level - complexity_level
                # Positive: GPU overpowered for model (excellent performance)
                # Zero: GPU well-matched to model (good performance)
                # Negative: GPU underpowered for model (poor performance)
                impact_score = power_level - complexity_level

                if impact_score >= 2:
                    level_name = 'luxury'  # Overpowered GPU, best performance
                elif impact_score == 1:
                    level_name = 'high'    # Strong GPU for the model
                elif impact_score == 0:
                    level_name = 'medium'  # Well-matched GPU-model pair
                elif impact_score == -1:
                    level_name = 'low'     # GPU slightly underpowered
                else:
                    level_name = 'low'     # GPU significantly underpowered

                min_impact, max_impact = impact_ranges[level_name]
                gamma_impact[j, k] = np.random.uniform(min_impact, max_impact)

        return gamma_impact

    def generate_error_impact(self, J: int, K: int, model_names: List[str],
                            gpu_tiers: List[str], P_gpu: np.ndarray) -> np.ndarray:
        """
        Generate error rate uncertainty impact factors ε[j,k] for routing-dependent uncertainty.

        Impact factor represents how GPU tier affects error rate uncertainty:
        - Higher values: Better GPUs provide more error reduction benefits
        - Lower values: GPU quality has less impact on error rates

        Depends on:
        - Model quality: More complex models benefit more from better GPUs
        - GPU precision: Higher precision (FP16) reduces errors vs quantized (INT4/INT8)

        Args:
            J: Number of base models
            K: Number of GPU configurations
            model_names: List of model names
            gpu_tiers: List of GPU tier names
            P_gpu: GPU compute powers

        Returns:
            2D array (J x K) with impact factors in [0, 1]
        """
        # More complex models have HIGHER error reduction impact on better hardware
        model_quality = {
            'Llama-3.2-1B': 'low',
            'Llama-3.2-3B': 'low',
            'Llama-3.1-8B': 'medium',
            'Llama-3.2-11B-Vision': 'medium',
            'Llama-3.1-70B': 'high',
            'Llama-3.2-31B': 'high'
        }

        def classify_gpu_precision(gpu_name, compute_power):
            # Precision penalty based on quantization
            if 'INT4' in gpu_name:
                precision_penalty = -1
            elif 'INT8' in gpu_name:
                precision_penalty = 0
            elif 'FP16' in gpu_name or 'BF16' in gpu_name:
                precision_penalty = 1
            else:
                precision_penalty = 1

            # Base level from compute power
            if compute_power < 150:
                base_level = 0
            elif compute_power < 400:
                base_level = 1
            elif compute_power < 900:
                base_level = 2
            else:
                base_level = 3

            final_level = max(0, min(3, base_level + precision_penalty))
            return ['low', 'medium', 'high', 'luxury'][final_level]

        impact_ranges = {
            'low': (0.2, 0.25),
            'medium': (0.40, 0.45),
            'high': (0.60, 0.65),
            'luxury': (0.80, 0.85)
        }

        error_impact = np.zeros((J, K))

        for j, model in enumerate(model_names):
            for k, gpu_name in enumerate(gpu_tiers):
                m_quality = model_quality[model]
                gpu_precision = classify_gpu_precision(gpu_name, P_gpu[k])

                quality_level = ['low', 'medium', 'high', 'luxury'].index(m_quality)
                precision_level = ['low', 'medium', 'high', 'luxury'].index(gpu_precision)

                # Weighted score: model quality has 2x weight
                weighted_score = (2 * quality_level + precision_level) / 3.0

                if weighted_score < 0.75:
                    level_name = 'low'
                elif weighted_score < 1.5:
                    level_name = 'medium'
                elif weighted_score < 2.25:
                    level_name = 'high'
                else:
                    level_name = 'luxury'

                min_impact, max_impact = impact_ranges[level_name]
                error_impact[j, k] = np.random.uniform(min_impact, max_impact)

        return error_impact

    # ==================== UTILITY FUNCTIONS ====================

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
        for k in [0, 5, 9]:  # Show sample: lowest, middle, highest
            print(f"  {data.gpu_tiers[k]:20s}: ${data.p_c[k]:.3f}")

        print(f"\n Constraints:")
        print(f"  Budget: ${data.delta}")
        print(f"  Storage: {data.C_storage} GB")
        print(f"  Delay Thresholds: {data.Delta_i}")
        print(f"  Error Thresholds: {data.epsilon}")

        # Print uncertainty info if available
        if isinstance(data, RobustLLMInferenceData):
            print(f"\n Uncertainty Parameters:")
            print(f"  Delay Budget (Γ_d): {data.Gamma_d}")
            print(f"  Error Budget (Γ_e): {data.Gamma_e}")
            print(f"  BigM Value: {data.BigM}")