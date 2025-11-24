# ICC Conference Project Structure & Relationships

## Directory Tree

```
ICC conference/
│
├── 📦 Core Optimization Models
│   ├── RODIU_LLM.py                          ⭐ Main robust optimization model
│   ├── RDDU_LLM_inference_opt.py             Earlier version
│   ├── LLM_DET.py                            Deterministic baseline
│   ├── robust_llm_optimization.py            Prototype
│   ├── deterministic_full_service.py         Full service variant
│   ├── llm_inference_with_tp.py              TP exploration
│   └── llm_workload_allocation.py            Basic allocation
│
├── 🔬 Sensitivity Analysis Scripts
│   ├── sensitivity_analysis_cost_budget.py           GPU cost × Budget
│   ├── sensitivity_analysis_delay_error_threshold.py Delay × Error
│   ├── sensitivity_analysis_gpu_cost_error.py        GPU cost × Error
│   └── sensitivity_analysis_memory_error.py          Memory × Error
│
├── 📊 Visualization & Analysis Tools
│   ├── error_sensitivity_analysis.py
│   ├── error_sensitivity_visualization.py
│   ├── plot_delay_error.py
│   ├── test_gamma_sensitivity.py
│   ├── visualize_tp_choice.py
│   ├── analyze_constraint_formulation.py
│   ├── constraint_analysis.py
│   └── print_gpu_tier_details.py
│
├── 📓 Jupyter Notebooks
│   ├── Experiment.ipynb                      Interactive experiments
│   └── Experiment_RO.ipynb                   Robust vs Deterministic
│
├── 🗂️ Data & Results
│   ├── sensitivity_results/                  (230 files)
│   │   ├── *.csv                            Numerical results
│   │   └── *.png                            Visualizations
│   ├── sensitivity_analysis/                 (8 CSV files)
│   └── plot/                                 (15 publication figures)
│       ├── *.jpg
│       ├── *.fig
│       └── sensitivity_plot_visulization.m
│
├── 📄 Documentation
│   ├── README.md
│   ├── ERROR_SENSITIVITY_ANALYSIS.md
│   └── ERROR_SENSITIVITY_README.md
│
├── 📑 Publication Materials
│   ├── ICC_submission.pdf                    Conference paper
│   ├── ICC_conference.pptx                   Presentation slides
│   ├── ICC_system_model.png                  System diagram
│   ├── system_model.jpg                      Alternative diagram
│   └── Latency_aware_Robust_LLM_...pdf       Preprint
│
├── ⚙️ Configuration
│   ├── requirements.txt                      Dependencies
│   ├── parameter_setup.py                    Default parameters
│   └── __pycache__/                          Python cache
│
└── 📈 Generated Data Files
    ├── delay_all_queries.png
    ├── delay_error_analysis.png
    ├── error_rate_all_queries.png
    ├── tp_degree_visualization.png
    ├── delay_sensitivity_results.csv
    ├── error_sensitivity_results.csv
    └── gamma_sensitivity_results.csv
```

---

## File Dependency Graph

```mermaid
graph TB
    subgraph Core["🎯 Core Model"]
        RODIU[RODIU_LLM.py<br/>⭐ Main Model]
        DG[DataGenerator]
        OPT[LLMInferenceOptimizer]
        DATA[LLMInferenceData]

        DG -->|generates| DATA
        DATA -->|input to| OPT
        RODIU -->|contains| DG
        RODIU -->|contains| OPT
    end

    subgraph Sensitivity["🔬 Sensitivity Analysis"]
        SA1[sensitivity_analysis_<br/>cost_budget.py]
        SA2[sensitivity_analysis_<br/>delay_error_threshold.py]
        SA3[sensitivity_analysis_<br/>gpu_cost_error.py]
        SA4[sensitivity_analysis_<br/>memory_error.py]
    end

    subgraph Visualization["📊 Visualization"]
        VIZ1[error_sensitivity_<br/>visualization.py]
        VIZ2[plot_delay_error.py]
        VIZ3[visualize_tp_choice.py]
        VIZ4[test_gamma_sensitivity.py]
    end

    subgraph Results["📁 Results"]
        CSV[*.csv files]
        PNG[*.png files]
        SENS_DIR[sensitivity_results/]
    end

    subgraph Notebooks["📓 Notebooks"]
        NB1[Experiment.ipynb]
        NB2[Experiment_RO.ipynb]
    end

    subgraph Legacy["📦 Legacy/Alternative"]
        RDDU[RDDU_LLM_inference_opt.py]
        DET[LLM_DET.py]
        ROBUST[robust_llm_optimization.py]
    end

    subgraph Publication["📑 Publication"]
        PAPER[ICC_submission.pdf]
        SLIDES[ICC_conference.pptx]
        FIGS[ICC_system_model.png]
    end

    %% Dependencies
    RODIU -->|imported by| SA1
    RODIU -->|imported by| SA2
    RODIU -->|imported by| SA3
    RODIU -->|imported by| SA4
    RODIU -->|imported by| NB1
    RODIU -->|imported by| NB2

    SA1 -->|generates| CSV
    SA2 -->|generates| CSV
    SA3 -->|generates| CSV
    SA4 -->|generates| CSV

    SA1 -->|generates| PNG
    SA2 -->|generates| PNG
    SA3 -->|generates| PNG
    SA4 -->|generates| PNG

    CSV -->|stored in| SENS_DIR
    PNG -->|stored in| SENS_DIR

    CSV -->|read by| VIZ1
    CSV -->|read by| VIZ2

    RODIU -->|used by| VIZ3
    RODIU -->|used by| VIZ4

    SENS_DIR -->|figures for| PAPER
    SENS_DIR -->|figures for| SLIDES

    style RODIU fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff
    style SA1 fill:#4ecdc4,stroke:#0a8f87,stroke-width:2px
    style SA2 fill:#4ecdc4,stroke:#0a8f87,stroke-width:2px
    style SA3 fill:#4ecdc4,stroke:#0a8f87,stroke-width:2px
    style SA4 fill:#4ecdc4,stroke:#0a8f87,stroke-width:2px
    style PAPER fill:#ffe66d,stroke:#f4a261,stroke-width:2px
```

---

## Execution Flow

```mermaid
flowchart TD
    START([Start Research]) --> GEN[Generate Base Data<br/>DataGenerator]

    GEN --> SINGLE{Single Run or<br/>Sensitivity?}

    SINGLE -->|Single| OPT[Run Optimizer<br/>build_and_solve_optimization_problem]
    SINGLE -->|Sensitivity| CHOOSE[Choose Analysis Type]

    OPT --> DISPLAY[Display Results<br/>display_results]
    DISPLAY --> ANALYZE[Manual Analysis]

    CHOOSE --> CB[Cost-Budget Analysis]
    CHOOSE --> DE[Delay-Error Analysis]
    CHOOSE --> GE[GPU-Error Analysis]
    CHOOSE --> ME[Memory-Error Analysis]

    CB --> LOOP1[Loop: 30 scenarios<br/>6 cost × 5 budget]
    DE --> LOOP2[Loop: 55 scenarios<br/>5 delay × 11 error]
    GE --> LOOP3[Loop: 121 scenarios<br/>11 cost × 11 error]
    ME --> LOOP4[Loop: scenarios<br/>memory × error]

    LOOP1 --> SCALE1[Scale Parameters:<br/>p_c, delta]
    LOOP2 --> SCALE2[Scale Parameters:<br/>Delta_i, epsilon]
    LOOP3 --> SCALE3[Scale Parameters:<br/>p_c, epsilon]
    LOOP4 --> SCALE4[Scale Parameters:<br/>memory, epsilon]

    SCALE1 --> RUN1[Run Optimization]
    SCALE2 --> RUN2[Run Optimization]
    SCALE3 --> RUN3[Run Optimization]
    SCALE4 --> RUN4[Run Optimization]

    RUN1 --> EXTRACT1[Extract Metrics]
    RUN2 --> EXTRACT2[Extract Metrics]
    RUN3 --> EXTRACT3[Extract Metrics]
    RUN4 --> EXTRACT4[Extract Metrics]

    EXTRACT1 --> CSV1[Save CSV]
    EXTRACT2 --> CSV2[Save CSV]
    EXTRACT3 --> CSV3[Save CSV]
    EXTRACT4 --> CSV4[Save CSV]

    CSV1 --> VIZ1[Generate Visualizations]
    CSV2 --> VIZ2[Generate Visualizations]
    CSV3 --> VIZ3[Generate Visualizations]
    CSV4 --> VIZ4[Generate Visualizations]

    VIZ1 --> SAVE1[Save Plots to<br/>sensitivity_results/]
    VIZ2 --> SAVE2[Save Plots to<br/>sensitivity_results/]
    VIZ3 --> SAVE3[Save Plots to<br/>sensitivity_results/]
    VIZ4 --> SAVE4[Save Plots to<br/>sensitivity_results/]

    SAVE1 --> NOTEBOOK[Optional: Jupyter<br/>Notebook Analysis]
    SAVE2 --> NOTEBOOK
    SAVE3 --> NOTEBOOK
    SAVE4 --> NOTEBOOK

    ANALYZE --> PAPER[Write Paper]
    NOTEBOOK --> PAPER

    PAPER --> SUBMIT[Submit to ICC]
    PAPER --> PRESENT[Create Presentation]

    style GEN fill:#4ecdc4,stroke:#0a8f87,stroke-width:2px
    style OPT fill:#ff6b6b,stroke:#c92a2a,stroke-width:2px
    style PAPER fill:#ffe66d,stroke:#f4a261,stroke-width:2px
    style SUBMIT fill:#95e1d3,stroke:#38ada9,stroke-width:2px
```

---

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph Input["📥 Input Parameters"]
        P1[Query Types: 6]
        P2[Models: 6]
        P3[GPU Tiers: 10]
        P4[TP Degrees: 4]
        P5[Costs, Thresholds<br/>Capacities]
    end

    subgraph DataGen["🔧 Data Generation"]
        DG[DataGenerator<br/>seed=42]
        LAMBDA[Arrival Rates]
        DELAYS[Processing Delays<br/>d_bar, d_hat]
        ERRORS[Error Rates<br/>e_bar, e_hat]
        COSTS[GPU Costs<br/>p_c]
    end

    subgraph Optimizer["⚙️ Optimization Engine"]
        MODEL[Gurobi MILP Model]
        VARS[Decision Variables:<br/>x, y, z, w, u]
        CONSTR[Constraints:<br/>Memory, Compute<br/>Budget, QoS]
        OBJ[Objective:<br/>Min C1+C2+C3+C4]
        ROBUST[Robust Formulation:<br/>Dual Variables τ, σ]
    end

    subgraph Output["📤 Optimization Output"]
        SOL[Solution Dict]
        X_OUT[Workload Allocation x]
        Y_OUT[GPU Provisioning y]
        W_OUT[TP Selection w]
        U_OUT[Unmet Demand u]
        COST_OUT[Cost Breakdown]
    end

    subgraph Analysis["📊 Analysis & Visualization"]
        METRICS[Extract Metrics:<br/>Cost, GPUs, Delay<br/>Error, Utilization]
        PLOTS[Generate Plots:<br/>Heatmaps, Trends<br/>Breakdowns]
        CSV_OUT[CSV Files]
        PNG_OUT[PNG Files]
    end

    P1 --> DG
    P2 --> DG
    P3 --> DG
    P4 --> DG
    P5 --> DG

    DG --> LAMBDA
    DG --> DELAYS
    DG --> ERRORS
    DG --> COSTS

    LAMBDA --> MODEL
    DELAYS --> MODEL
    ERRORS --> MODEL
    COSTS --> MODEL

    MODEL --> VARS
    VARS --> CONSTR
    VARS --> OBJ
    CONSTR --> ROBUST
    OBJ --> ROBUST

    ROBUST --> SOL
    SOL --> X_OUT
    SOL --> Y_OUT
    SOL --> W_OUT
    SOL --> U_OUT
    SOL --> COST_OUT

    X_OUT --> METRICS
    Y_OUT --> METRICS
    W_OUT --> METRICS
    U_OUT --> METRICS
    COST_OUT --> METRICS

    METRICS --> PLOTS
    METRICS --> CSV_OUT
    PLOTS --> PNG_OUT

    style DG fill:#4ecdc4,stroke:#0a8f87,stroke-width:2px
    style MODEL fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px
    style METRICS fill:#95e1d3,stroke:#38ada9,stroke-width:2px
    style CSV_OUT fill:#ffe66d,stroke:#f4a261,stroke-width:2px
    style PNG_OUT fill:#ffe66d,stroke:#f4a261,stroke-width:2px
```

---

## Module Relationship Matrix

| Module | RODIU_LLM.py | Sensitivity Scripts | Visualization Tools | Notebooks | Results Dir |
|--------|--------------|---------------------|---------------------|-----------|-------------|
| **RODIU_LLM.py** | - | ✅ Imported | ✅ Imported | ✅ Imported | ❌ |
| **Sensitivity Scripts** | ✅ Imports | - | ❌ | ❌ | ✅ Writes |
| **Visualization Tools** | ✅ Some import | ❌ | - | ❌ | ✅ Reads/Writes |
| **Notebooks** | ✅ Imports | ❌ | ❌ | - | ✅ Reads/Writes |
| **Results Dir** | ❌ | ✅ Read from | ✅ Read from | ✅ Read from | - |

---

## Key File Relationships

### 1️⃣ Core Model Dependencies
```
RODIU_LLM.py
    ├── gurobipy (Gurobi optimizer)
    ├── numpy (numerical arrays)
    ├── pandas (data structures)
    └── dataclasses (data containers)
```

### 2️⃣ Sensitivity Analysis Pattern
```
sensitivity_analysis_*.py
    ├── RODIU_LLM.DataGenerator
    ├── RODIU_LLM.LLMInferenceOptimizer
    ├── matplotlib.pyplot
    ├── seaborn
    └── Output:
        ├── sensitivity_results/{name}_{timestamp}.csv
        └── sensitivity_results/{plot_type}_{timestamp}.png
```

### 3️⃣ Visualization Dependencies
```
plot_delay_error.py
    └── Generates:
        ├── delay_all_queries.png
        ├── delay_error_analysis.png
        └── error_rate_all_queries.png

visualize_tp_choice.py
    └── Generates:
        └── tp_degree_visualization.png

error_sensitivity_visualization.py
    ├── Reads: error_sensitivity_results.csv
    └── Generates: error analysis plots
```

### 4️⃣ Publication Pipeline
```
RODIU_LLM.py
    └── Sensitivity Analysis
        └── sensitivity_results/*.png
            └── plot/ (curated figures)
                └── ICC_submission.pdf
                └── ICC_conference.pptx
```

---

## Typical Usage Patterns

### Pattern 1: Single Optimization Run
```bash
python RODIU_LLM.py
# Output: Console display of results
```

### Pattern 2: Sensitivity Analysis
```bash
python sensitivity_analysis_cost_budget.py
# Output:
#   - sensitivity_results/sensitivity_cost_budget_{timestamp}.csv
#   - sensitivity_results/heatmaps_cost_budget_{timestamp}.png
#   - sensitivity_results/cost_trends_cost_budget_{timestamp}.png
#   - (5+ visualization files)
```

### Pattern 3: Interactive Exploration
```bash
jupyter notebook Experiment.ipynb
# Imports RODIU_LLM, runs custom scenarios
```

### Pattern 4: Generate Publication Figures
```bash
python plot_delay_error.py
python visualize_tp_choice.py
# Output: High-quality figures in current directory
# Then manually move to plot/ directory
```

---

## File Size Distribution

| Category | Files | Total Size |
|----------|-------|------------|
| **Core Python** | 20 files | ~500 KB |
| **Notebooks** | 2 files | ~4.4 MB |
| **PDFs** | 3 files | ~4.5 MB |
| **Presentation** | 1 file | ~2.4 MB |
| **Images** | ~250 files | ~50+ MB |
| **Results CSV** | ~100 files | ~5 MB |
| **Documentation** | 3 MD files | ~50 KB |
| **Total** | ~380 files | ~65+ MB |

---

## Version History (Inferred)

```
Version 1: llm_workload_allocation.py
    └── Basic workload allocation, no TP, no robustness

Version 2: robust_llm_optimization.py
    └── Added robust optimization, but no TP

Version 3: llm_inference_with_tp.py
    └── Explored tensor parallelism

Version 4: RDDU_LLM_inference_opt.py
    └── Combined robust + TP, early version

Version 5: RODIU_LLM.py ⭐ CURRENT
    └── Refined model with decision-dependent uncertainty
    └── Full constraint analysis
    └── Production-ready code
```

---

## Quick Reference

### 🎯 Want to run the main model?
→ `python RODIU_LLM.py`

### 🔬 Want to do sensitivity analysis?
→ `python sensitivity_analysis_*.py`

### 📊 Want to visualize existing results?
→ Check `sensitivity_results/` directory

### 📓 Want to experiment interactively?
→ `jupyter notebook Experiment.ipynb`

### 📄 Want to see research output?
→ `ICC_submission.pdf` and `ICC_conference.pptx`

### 🐛 Want to debug constraints?
→ `python constraint_analysis.py` or `analyze_constraint_formulation.py`

---

**Last Updated**: Based on file timestamps up to November 10, 2025
