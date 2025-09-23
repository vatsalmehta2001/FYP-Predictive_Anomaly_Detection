# Experiment Management & MLflow Organization

This document outlines the experiment tracking strategy, naming conventions, and MLflow organization for systematic research management in the FYP Energy Forecasting project.

## MLflow Experiment Taxonomy

### Experiment Categories

Our experiments are organized into four primary categories, each addressing different aspects of the research:

#### 1. Baseline Experiments (`baseline/*`)
Traditional forecasting approaches for comparison and validation.

**Sub-categories:**
- `baseline/statistical` - Classical time series methods (ARIMA, ETS, STL)
- `baseline/ml` - Traditional ML approaches (Random Forest, XGBoost, SVR)  
- `baseline/dl` - Standard deep learning models (LSTM, GRU, Transformer)
- `baseline/pretrained` - Large pre-trained models (TimeGPT, Lag-Llama)

**Example Run Names:**
- `baseline/statistical/arima_ukdale_h1_seasonal`
- `baseline/dl/lstm_lcl_attention_dropout`
- `baseline/pretrained/timegpt_ensemble_quantiles`

#### 2. Custom Architecture Experiments (`custom/*`)
Novel architectures adapted for energy forecasting.

**Sub-categories:**
- `custom/patchtst` - PatchTST variants with energy-specific modifications
- `custom/nbeats` - N-BEATS adaptations with interpretable blocks
- `custom/hybrid` - Combined statistical-neural approaches
- `custom/uncertainty` - Uncertainty quantification methods

**Example Run Names:**
- `custom/patchtst/energy_aware_patches_30min`
- `custom/nbeats/seasonal_trend_decomp_blocks`
- `custom/uncertainty/monte_carlo_dropout_quantiles`

#### 3. Self-Play Experiments (`selfplay/*`)
Core research contribution: propose→solve→verify training.

**Sub-categories:**
- `selfplay/proposer` - Scenario generation model experiments
- `selfplay/solver` - Forecasting model in self-play loop
- `selfplay/verifier` - Constraint validation and reward modeling
- `selfplay/integrated` - Full propose→solve→verify training runs

**Example Run Names:**
- `selfplay/proposer/ev_spike_scenario_generator`
- `selfplay/solver/patchtst_selfplay_trained`
- `selfplay/verifier/physics_constraint_rewards`
- `selfplay/integrated/full_loop_v2_1000_episodes`

#### 4. Ablation Studies (`ablation/*`)
Systematic component analysis and sensitivity studies.

**Sub-categories:**
- `ablation/architecture` - Model component importance
- `ablation/features` - Feature contribution analysis  
- `ablation/hyperparams` - Hyperparameter sensitivity
- `ablation/rewards` - Self-play reward function components

**Example Run Names:**
- `ablation/architecture/attention_heads_1_2_4_8`
- `ablation/features/weather_vs_calendar_vs_lags`
- `ablation/rewards/physics_vs_statistical_constraints`

## Experiment Naming Conventions

### Run Name Structure
```
{category}/{subcategory}/{model}_{dataset}_{variant}_{seed}
```

### Component Definitions
- **Category**: One of `baseline`, `custom`, `selfplay`, `ablation`
- **Subcategory**: Specific research area within category
- **Model**: Architecture or method name (abbreviated)
- **Dataset**: Data source (`ukdale`, `lcl`, `combined`)
- **Variant**: Specific configuration or modification
- **Seed**: Random seed for reproducibility (optional for deterministic runs)

### Naming Examples
```
baseline/dl/lstm_ukdale_bidirectional_42
custom/patchtst/energy_patches_lcl_uncertainty_123
selfplay/integrated/propose_solve_verify_combined_v3_456
ablation/features/no_weather_ukdale_comparison_789
```

### Tag Conventions
Experiments are tagged with metadata for easy filtering:

**Common Tags:**
- `dataset:ukdale` / `dataset:lcl` / `dataset:ssen` / `dataset:combined`
- `horizon:1h` / `horizon:6h` / `horizon:24h` / `horizon:168h`
- `resolution:30min` / `resolution:1h` / `resolution:1d`
- `uncertainty:point` / `uncertainty:quantile` / `uncertainty:distributional`
- `season:summer` / `season:winter` / `season:all`
- `status:running` / `status:completed` / `status:failed` / `status:cancelled`

**Research-Specific Tags:**
- `selfplay:proposer_only` / `selfplay:solver_only` / `selfplay:full_loop`
- `baseline:comparison` / `baseline:ablation`
- `eval:feeder_validation` / `eval:household_accuracy`

## Artifact Management

### Required Artifacts

Every experiment run must log the following artifacts:

#### 1. Configuration Files
```python
# Log all configuration
mlflow.log_dict(config, "config.yaml")
mlflow.log_dict(model_params, "model_config.yaml") 
mlflow.log_dict(data_params, "data_config.yaml")
```

#### 2. Model Checkpoints
```python
# Save model at best validation performance
mlflow.pytorch.log_model(model, "best_model")
mlflow.log_artifact("model_checkpoint.pth")
```

#### 3. Metrics and Plots
```python
# Core metrics (logged every epoch)
mlflow.log_metrics({
    "mae_validation": mae_val,
    "mape_validation": mape_val, 
    "rmse_validation": rmse_val,
    "pinball_loss": pinball_loss,  # for quantile models
    "coverage_80": coverage_80,    # for uncertainty quantification
})

# Summary plots
mlflow.log_figure(forecast_plot, "forecasts.png")
mlflow.log_figure(residual_plot, "residuals.png")
mlflow.log_figure(uncertainty_plot, "uncertainty.png")
```

#### 4. Evaluation Reports
```python
# Comprehensive evaluation
mlflow.log_dict(evaluation_metrics, "metrics.json")
mlflow.log_text(evaluation_summary, "evaluation_report.md")
```

### Artifact Organization
```
mlruns/
├── {experiment_id}/
│   ├── {run_id}/
│   │   ├── artifacts/
│   │   │   ├── config.yaml              # Experiment configuration
│   │   │   ├── model_config.yaml        # Model hyperparameters
│   │   │   ├── data_config.yaml         # Data processing parameters
│   │   │   ├── best_model/              # Model checkpoint directory
│   │   │   ├── metrics.json             # Final evaluation metrics
│   │   │   ├── evaluation_report.md     # Human-readable summary
│   │   │   ├── forecasts.png            # Prediction visualizations
│   │   │   ├── residuals.png            # Error analysis plots
│   │   │   ├── uncertainty.png          # Uncertainty quantification
│   │   │   └── feature_importance.png   # Feature analysis (if applicable)
│   │   ├── metrics/                     # Time series metrics
│   │   ├── params/                      # Logged parameters
│   │   └── tags/                        # Experiment tags
```

## Metrics Definition

### Core Forecasting Metrics
```python
CORE_METRICS = {
    # Point forecasting accuracy
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
    "rmse": root_mean_squared_error,
    "smape": symmetric_mean_absolute_percentage_error,
    
    # Distribution accuracy (for uncertainty models)
    "pinball_loss_10": lambda y, q: pinball_loss(y, q, 0.1),
    "pinball_loss_50": lambda y, q: pinball_loss(y, q, 0.5),
    "pinball_loss_90": lambda y, q: pinball_loss(y, q, 0.9),
    
    # Coverage metrics
    "coverage_50": lambda y, q_low, q_high: coverage_score(y, q_low, q_high),
    "coverage_80": lambda y, q_low, q_high: coverage_score(y, q_low, q_high),
    "coverage_95": lambda y, q_low, q_high: coverage_score(y, q_low, q_high),
    
    # Energy-specific metrics
    "peak_mae": peak_mean_absolute_error,      # Error during peak hours
    "daily_energy_mape": daily_energy_mape,    # Daily total energy accuracy
    "ramp_rate_error": ramp_rate_error,        # Gradient prediction accuracy
}
```

### Self-Play Specific Metrics
```python
SELFPLAY_METRICS = {
    # Proposer evaluation
    "scenario_diversity": scenario_diversity_score,
    "scenario_realism": physics_constraint_score,
    "scenario_difficulty": solver_error_distribution,
    
    # Solver improvement
    "solve_rate": proportion_scenarios_solved,
    "improvement_rate": accuracy_improvement_per_episode,
    "generalization": cross_scenario_performance,
    
    # Verifier accuracy  
    "constraint_precision": constraint_violation_detection,
    "reward_correlation": reward_ground_truth_correlation,
    "false_positive_rate": invalid_scenario_acceptance,
}
```

### Feeder Validation Metrics
```python
FEEDER_METRICS = {
    # Distributional comparison with SSEN
    "ks_statistic": kolmogorov_smirnov_test,
    "wasserstein_distance": wasserstein_distance,
    "jensen_shannon_divergence": js_divergence,
    
    # Peak load analysis
    "peak_bias": mean_peak_load_difference,
    "peak_correlation": peak_time_correlation,
    "load_factor_error": load_factor_comparison,
    
    # Temporal pattern matching
    "daily_profile_similarity": daily_profile_correlation,
    "seasonal_pattern_match": seasonal_decomposition_similarity,
    "holiday_effect_match": holiday_impact_correlation,
}
```

## Experiment Lifecycle

### 1. Experiment Design Phase
```python
# Create experiment and log design decisions
experiment_id = mlflow.create_experiment("selfplay/integrated/v1_0")
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_dict(experiment_design, "design_document.yaml")
    mlflow.log_text(hypothesis, "hypothesis.md")
    mlflow.set_tags({
        "research_phase": "design",
        "hypothesis": "self_play_improves_generalization",
        "expected_duration": "1_week"
    })
```

### 2. Execution Phase
```python
# Log progress and intermediate results
with mlflow.start_run(run_name="selfplay_full_v1_0_seed42"):
    for episode in range(num_episodes):
        # Training step
        metrics = train_episode()
        mlflow.log_metrics(metrics, step=episode)
        
        # Periodic checkpointing
        if episode % 100 == 0:
            mlflow.pytorch.log_model(model, f"checkpoint_ep{episode}")
```

### 3. Evaluation Phase
```python
# Comprehensive evaluation and reporting
evaluation_results = evaluate_model(model, test_data)
mlflow.log_dict(evaluation_results, "final_evaluation.json")

# Generate visualizations
create_forecast_plots()
create_residual_analysis()
create_uncertainty_plots()

# Log all artifacts
mlflow.log_artifacts("plots/", "evaluation_plots")
```

### 4. Analysis and Documentation
```python
# Generate automated report
report = generate_experiment_report(run_id)
mlflow.log_text(report, "experiment_report.md")

# Compare with baselines
comparison = compare_with_baselines(current_results, baseline_results)
mlflow.log_dict(comparison, "baseline_comparison.json")
```

## Experiment Comparison & Analysis

### Cross-Experiment Comparison
```python
# Compare multiple experiments
def compare_experiments(experiment_names, metric="mae"):
    """Compare performance across different experiment types."""
    results = {}
    for exp_name in experiment_names:
        runs = mlflow.search_runs(experiment_ids=[get_experiment_id(exp_name)])
        results[exp_name] = runs[metric].min()
    return results

# Statistical significance testing
def statistical_comparison(baseline_runs, treatment_runs, alpha=0.05):
    """Perform statistical significance test between experiment groups."""
    return welch_ttest(baseline_runs, treatment_runs, alpha)
```

### Automated Reporting
```python
# Generate periodic research summaries
def generate_research_summary(date_range):
    """Create automated summary of recent experiments."""
    summary = {
        "experiments_completed": count_completed_experiments(date_range),
        "best_performers": get_top_performers_by_category(),
        "failed_experiments": get_failed_experiments_with_reasons(),
        "resource_usage": get_compute_resource_summary(),
        "next_experiments": get_planned_experiments()
    }
    return summary
```

## Best Practices

### 1. Reproducibility
- Always set and log random seeds
- Pin all dependency versions in environment
- Log complete configuration for every run
- Use deterministic algorithms where possible

### 2. Resource Management
- Log compute resource usage (GPU hours, memory)
- Use early stopping to prevent resource waste
- Implement automatic cleanup of failed runs
- Monitor and alert on long-running experiments

### 3. Collaboration
- Use descriptive run names and tags
- Document experimental hypotheses upfront
- Share interesting results through experiment reports
- Regular team reviews of experiment progress

### 4. Quality Control
- Validate data integrity before training
- Implement sanity checks for model outputs
- Log warnings and errors for debugging
- Regular audits of experiment metadata quality
