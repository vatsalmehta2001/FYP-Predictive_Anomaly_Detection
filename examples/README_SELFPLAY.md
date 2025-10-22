# Grid Guardian Self-Play Examples

This directory contains demonstration scripts for the Grid Guardian self-play reinforcement learning system.

---

## Quick Start

### 1. Basic Self-Play Demo (5 minutes)

Validates the core propose-solve-verify loop with minimal training:

```bash
python examples/selfplay_quick_demo.py
```

**Expected Output**:
- ✅ 5 episodes complete in <1 second
- ✅ MAE ~0.5-1.5 kWh (with PyTorch) or NaN (fallback)
- ✅ Verification reward >0 (physics compliant)
- ✅ Metrics plot: `docs/figures/selfplay_demo_metrics.png`

**Use Case**: Quick smoke test to verify system functionality.

---

### 2. BDH-Enhanced Demo (10 minutes)

Demonstrates Dragon Hatchling-inspired enhancements with extended training:

```bash
python examples/selfplay_bdh_demo.py
```

**Expected Output**:
- ✅ 20 episodes with Hebbian constraint adaptation
- ✅ Graph-based scenario sampling (non-uniform distribution)
- ✅ Comprehensive BDH metrics plot: `docs/figures/selfplay_bdh_metrics.png`
- ✅ Constraint weight statistics
- ✅ Scenario transition analysis

**Use Case**: Validate BDH enhancements and analyze adaptive behavior.

---

## BDH Enhancements Reference

### What is BDH?

The **Dragon Hatchling (BDH)** is a biologically-inspired neural network architecture that bridges Transformers and brain models using:

- **Synaptic plasticity**: Connections strengthen with co-activation (Hebbian learning)
- **Modular graph structure**: Neurons form scale-free networks with high clustering
- **Sparse activations**: ~5% sparsity enables interpretability

**Paper**: Kosowski et al. (2025). *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain*. [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)

### How Grid Guardian Uses BDH Concepts

We integrate **lightweight BDH concepts** without replacing PatchTST:

| BDH Concept                  | Grid Guardian Application                                  | Implementation                     |
|------------------------------|------------------------------------------------------------|------------------------------------|
| Synaptic plasticity (σ)      | Constraint weights adapt when violated                     | `HebbianVerifier`                  |
| Modular neuron graph         | Scenarios follow causal relationships (graph sampling)     | `GraphBasedProposer`               |
| Sparse activations (~5%)     | Monitor hidden state sparsity (future work)                | `SparseActivationMonitor`          |
| Heavy-tailed degree dist.    | Scenario graph with uneven connections                     | Transition probability matrix      |

**Why Not Full BDH Architecture?**
- PatchTST is proven for time-series forecasting
- BDH is very recent (Sep 2025), limited validation
- Hybrid approach gives best of both: Transformer performance + brain-inspired adaptation

---

## File Descriptions

### `selfplay_quick_demo.py` (162 lines)

**Purpose**: Fast validation of core self-play system.

**Key Features**:
- Synthetic data generation (no LCL dependency)
- 5-episode training loop
- 4-panel metrics visualization
- Success criteria checks

**When to Use**: 
- After code changes (regression testing)
- Quick smoke test
- Teaching/demonstration

---

### `selfplay_bdh_demo.py` (331 lines)

**Purpose**: Comprehensive BDH enhancement demonstration.

**Key Features**:
- 20-episode training with BDH enhancements
- Hebbian weight adaptation tracking
- Graph-based scenario analysis
- 9-panel comprehensive visualization
- Scenario distribution analysis

**When to Use**:
- Validate BDH integration
- Analyze adaptive behavior
- Generate plots for papers/presentations
- Compare to baseline (run both demos)

---

## Advanced Usage

### Custom BDH Configuration

```python
from fyp.selfplay import ProposerAgent, SolverAgent, VerifierAgent
from fyp.selfplay.bdh_enhancements import create_bdh_enhanced_trainer

# Initialize components
proposer = ProposerAgent(...)
solver = SolverAgent(...)
verifier = VerifierAgent(...)

# Create BDH-enhanced trainer with custom settings
trainer = create_bdh_enhanced_trainer(
    base_proposer=proposer,
    base_solver=solver,
    base_verifier=verifier,
    config={'alpha': 0.2, 'batch_size': 8},
    enable_hebbian=True,      # Constraint adaptation
    enable_graph=True,         # Scenario graph
    enable_sparsity=False,     # Requires model changes
)

# Train
metrics = trainer.train(num_episodes=100)

# Analyze Hebbian adaptation
if hasattr(trainer, 'hebbian_verifier'):
    stats = trainer.hebbian_verifier.get_weight_statistics()
    for constraint, info in stats.items():
        print(f"{constraint}: {info['weight']:.3f} "
              f"(violations: {info['activation_rate']:.1%})")

# Analyze graph structure
if hasattr(trainer, 'graph_proposer'):
    graph_stats = trainer.graph_proposer.get_graph_statistics()
    print(f"Graph density: {graph_stats['graph_density']:.2%}")
```

### Ablation Study: BDH vs Baseline

```python
# Baseline (no BDH)
baseline_trainer = SelfPlayTrainer(proposer, solver, verifier, config)
baseline_metrics = baseline_trainer.train(num_episodes=50)

# BDH-enhanced
bdh_trainer = create_bdh_enhanced_trainer(
    proposer, solver, verifier, config,
    enable_hebbian=True,
    enable_graph=True
)
bdh_metrics = bdh_trainer.train(num_episodes=50)

# Compare
baseline_mae = baseline_metrics[-1]['avg_mae']
bdh_mae = bdh_metrics[-1]['avg_mae']
improvement = (baseline_mae - bdh_mae) / baseline_mae * 100
print(f"MAE improvement: {improvement:.1f}%")
```

---

## Expected Results

### With PyTorch + PatchTST

| Metric                     | Quick Demo (5 ep) | BDH Demo (20 ep) | Full Training (100 ep) |
|----------------------------|-------------------|------------------|------------------------|
| Final MAE                  | 0.8-1.2 kWh       | 0.5-0.8 kWh      | 0.3-0.6 kWh            |
| Verification Reward        | 0.00-0.05         | 0.02-0.08        | 0.05-0.15              |
| Scenario Diversity         | 60-80%            | 80-100%          | 90-100%                |
| Constraint Violations      | <5%               | <2%              | <1%                    |
| Training Time              | <1 second         | ~5 seconds       | ~2 minutes             |

### Fallback Mode (No PyTorch)

| Metric                     | Quick Demo        | BDH Demo         | Notes                          |
|----------------------------|-------------------|------------------|--------------------------------|
| Final MAE                  | NaN               | NaN              | Fallback returns last value    |
| Verification Reward        | 0.00-0.05         | 0.02-0.08        | Still validates physics        |
| Solver Loss                | 1.000 (constant)  | 1.000 (constant) | No gradient updates            |
| Training Time              | <1 second         | <1 second        | Faster (no GPU)                |

**Recommendation**: Install PyTorch for meaningful metrics:
```bash
poetry install  # or: pip install torch
```

---

## Troubleshooting

### Issue: Import Error

```python
ModuleNotFoundError: No module named 'fyp'
```

**Solution**:
```bash
export PYTHONPATH="$(pwd)/src"
# or run from project root:
python -m examples.selfplay_quick_demo
```

---

### Issue: PyTorch Not Found

```
PyTorch not available. Some solver functionality will be limited.
PatchTST not available. Solver will use fallback methods.
```

**Solution**: Install dependencies:
```bash
poetry install
# or: pip install torch transformers
```

---

### Issue: JSON Key Error

```
KeyError: 'min_lagging'
```

**Solution**: Ensure `data/derived/ssen_constraints.json` exists:
```bash
# Generate constraints if missing:
python src/fyp/ingestion/generate_ssen_constraints.py
```

---

### Issue: NaN MAE

**Cause**: Using fallback solver (no PyTorch/PatchTST)

**Solution**: Either:
1. Install PyTorch (recommended)
2. Ignore MAE metrics (focus on verification rewards)

---

## Next Steps

1. **Run both demos** to ensure system works
2. **Install PyTorch** for full functionality
3. **Train on LCL data** (see `docs/selfplay_implementation.md`)
4. **Benchmark baselines** (see `src/fyp/baselines/`)
5. **Read validation report** (`docs/selfplay_validation_report.md`)

---

## References

1. **BDH Paper**:  
   Kosowski et al. (2025). *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain*.  
   arXiv:2509.26507. [https://arxiv.org/abs/2509.26507](https://arxiv.org/abs/2509.26507)

2. **Grid Guardian Docs**:
   - `docs/selfplay_design.md`: Architecture overview
   - `docs/selfplay_implementation.md`: Implementation details
   - `docs/selfplay_validation_report.md`: Validation results

3. **Related Work**:
   - PatchTST: Nie et al. (2023)
   - Self-play RL: Silver et al. (2017) - AlphaGo Zero
   - Physics-informed ML: Raissi et al. (2019)

---

**Last Updated**: October 22, 2025  
**Maintainer**: Grid Guardian Team

