# Next Sprint Roadmap

This document outlines the key development priorities for the next development sprint, focusing on advancing the core research contributions and operational capabilities.

## Priority Issues for Next Sprint

### 🎮 Self-Play v1 Implementation
**Epic**: Core research contribution - implement the propose→solve→verify self-play loop

**Scope**:
- **Proposer Module**: Scenario generation for EV spikes, cold snaps, peak shifts, blackouts, and missing data events
- **Solver Integration**: Adapt PatchTST/baseline forecasters to work within self-play training loop
- **Verifier Component**: Physics-based constraint validation with realistic reward signals
- **Training Loop**: Episode management with verifiable reward feedback and model improvement tracking
- **Logging & Visualization**: MLflow integration for self-play metrics, scenario diversity, and solver improvement curves

**Acceptance Criteria**:
- Complete propose→solve→verify training loop functional on sample data
- Verifier enforces non-negativity, ramp limits, energy budgets, and temporal plausibility
- MLflow tracks episode progress, scenario generation quality, and solver improvement
- Self-play models outperform baselines on challenging synthetic scenarios

**Estimated Effort**: 2-3 weeks

---

### 🔄 SSEN Sync Polish
**Epic**: Production-grade SSEN data synchronization with incremental updates

**Scope**:
- **Incremental Sync**: `--since` date parameter for fetching only new/updated data
- **Idempotent Partitions**: Date-based partitioning that allows safe re-runs without duplication
- **Sync Reporting**: JSON report with sync statistics, data quality metrics, and API response summaries
- **Error Recovery**: Robust handling of API failures, partial responses, and data validation errors
- **Cache Management**: Intelligent cache invalidation and cleanup policies

**Features**:
```bash
# Incremental sync from last week
python -m fyp.ingestion.cli ssen --since 2025-01-01 --sync-report

# Full sync with quality report
python -m fyp.ingestion.cli ssen --force-refresh --quality-report
```

**Acceptance Criteria**:
- Incremental sync fetches only new data since specified date
- Sync report JSON includes data quality metrics, API performance, and error summaries
- Idempotent operations allow safe re-execution without data duplication
- Cache management keeps disk usage reasonable while maintaining performance

**Estimated Effort**: 1-2 weeks

---

### 📊 Rolling Backtests + MLflow Leaderboard
**Epic**: Systematic model evaluation with time-series cross-validation

**Scope**:
- **Rolling Backtest Framework**: Time-series cross-validation with expanding/sliding windows
- **Comprehensive Metrics**: OWA (Overall Weighted Average), CRPS (Continuous Ranked Probability Score), precision/recall curves
- **MLflow Tables**: Leaderboard comparing all models (baselines, PatchTST, self-play) across datasets
- **Performance Analytics**: Model ranking, statistical significance testing, and performance attribution
- **Latency Analysis**: Real-time anomaly detection responsiveness measurement

**Features**:
```bash
# Rolling backtest evaluation
python -m fyp.evaluation.backtest --models all --datasets all --horizons 24,48,168

# Generate leaderboard
python -m fyp.evaluation.leaderboard --metric owa --export-html
```

**Deliverables**:
- **OWA Metrics**: Industry-standard forecasting evaluation across multiple horizons
- **CRPS Scoring**: Probabilistic forecast evaluation for uncertainty quantification
- **PR/Latency Tables**: Anomaly detection performance with response time analysis
- **MLflow Dashboard**: Interactive model comparison with statistical significance tests

**Acceptance Criteria**:
- Rolling backtest validates temporal generalization across all models
- Leaderboard ranks models by OWA, CRPS, and detection metrics
- Statistical significance testing identifies truly superior approaches
- HTML export enables easy sharing of evaluation results

**Estimated Effort**: 2-3 weeks

---

## Implementation Notes

### Dependencies
- **Self-Play**: Requires reward function design and scenario parameterization
- **SSEN Sync**: Depends on API availability and rate limiting considerations  
- **Backtests**: Builds on existing model implementations and metrics framework

### Success Metrics
- **Self-Play**: Solver improvement over episodes, scenario diversity scores, constraint satisfaction rates
- **SSEN Sync**: Sync reliability, API efficiency, data freshness metrics
- **Backtests**: Statistical confidence in model rankings, evaluation completeness

### Risk Mitigation
- **Self-Play Complexity**: Start with simple scenarios, gradually increase sophistication
- **API Dependencies**: Implement robust fallbacks and offline modes for SSEN sync
- **Evaluation Scale**: Begin with subset of models/datasets, expand systematically

These priorities advance both the core research innovation (self-play) and operational excellence (sync robustness, evaluation rigor) needed for a successful Final Year Project.
