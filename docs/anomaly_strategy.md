# Anomaly Detection Strategy

## Problem Statement

Our datasets (UK-DALE, LCL, SSEN) lack ground-truth anomaly labels. This is not a limitation but an opportunity to demonstrate unsupervised and physics-informed approaches.

## Core Challenge

Traditional supervised anomaly detection requires labeled data:
- Normal vs. Anomalous consumption examples
- Manually annotated events
- Expensive expert labeling

**Our datasets do NOT have this.** Instead, we have:
- High-quality consumption time series
- Appliance-level disaggregation (UK-DALE)
- Real feeder constraints (SSEN)
- Large household sample (LCL: 5,567 households)

## Solution: Three-Tier Approach

### Tier 1: Physics-Based Constraints (SSEN Validation)

Anomalies that **violate physical network constraints**:

**From SSEN feeder data:**
1. **Voltage violations**: Outside 207-253V (UK statutory ±10%)
2. **Capacity violations**: Load exceeds transformer rating
3. **Power factor violations**: Outside 0.8-1.0 range
4. **Impossible patterns**: 
   - Negative consumption
   - Zero consumption for >48 hours (occupied household)
   - Instantaneous load >100kW (typical household max ~10-15kW)

**Implementation:**
```python
def is_physically_impossible(consumption, metadata):
    """Hard constraints - binary decision"""
    if consumption < 0:
        return True, "Negative consumption"
    if consumption > metadata['household_max_capacity']:
        return True, "Exceeds physical capacity"
    # ... more checks
    return False, None
```

### Tier 2: Statistical Anomalies (Learned Baselines)

Anomalies that are **statistically improbable**:

1. **Point anomalies**: Single readings >5σ from rolling mean
2. **Contextual anomalies**: Normal value at wrong time (high consumption at 3 AM)
3. **Collective anomalies**: Unusual sequence (flat-line for 6 hours during peak times)

**Baseline detectors (already implemented):**
- Decomposition-based (trend + seasonal + residual analysis)
- Statistical (z-score with adaptive thresholds)
- Isolation Forest (for comparison)

### Tier 3: Self-Play Learned Anomalies

**This is our novel contribution.**

The self-play verifier learns what consumption patterns are "plausible" given:
- Historical household behavior
- Seasonal context
- SSEN network constraints
- Cross-household patterns

**Proposer** generates consumption scenarios →  
**Solver** forecasts or detects →  
**Verifier** scores based on:
- Physics constraints (Tier 1)
- Statistical likelihood (Tier 2)  
- Learned plausibility (updated through self-play)

## Evaluation Strategy

### Quantitative Evaluation (Synthetic Labels)

**Create test set with injected anomalies:**
```python
# Injection types
anomaly_types = {
    'spike': lambda x: x * np.random.uniform(3, 7),
    'dropout': lambda x: 0,
    'shift': lambda x: x + np.random.normal(0, 3*x.std()),
    'flatline': lambda x: x.mean(),
}

# Inject into 10% of test data
# Evaluate: Precision, Recall, F1, ROC-AUC
```

**Metrics:**
- Precision: % of flagged anomalies that are truly injected
- Recall: % of injected anomalies that are flagged
- F1-Score: Harmonic mean
- ROC-AUC: Overall discrimination ability

### Qualitative Evaluation (SSEN Validation)

**On real LCL data:**
1. Flag top 1% most anomalous windows
2. Verify against SSEN constraints
3. Manual inspection with domain knowledge
4. Document false positive/negative patterns

### Comparison Baselines

Compare our self-play approach against:
1. Decomposition detector (unsupervised)
2. Statistical detector (unsupervised)
3. Isolation Forest (unsupervised)
4. Autoencoder (neural, unsupervised)

**Success criteria:**
- Match or exceed baseline F1 on synthetic test set
- Lower false positive rate on real data
- Validate against SSEN constraints

## Implementation Roadmap

**Phase 1 (Current):** Data preparation
- Ingestion pipelines working
- Exploratory analysis complete
- Constraints documented

**Phase 2 (Next 2 weeks):** Baseline evaluation
- Run existing detectors on LCL
- Create synthetic test set
- Establish baseline performance

**Phase 3 (Weeks 3-6):** Self-play implementation
- Build proposer agent
- Implement verifier with SSEN constraints
- Train self-play loop
- Evaluate against baselines

**Phase 4 (Weeks 7-8):** Results & writing
- Final evaluation
- Visualizations
- Dissertation writing

## Constraint Definitions (From SSEN Metadata)

### Voltage Constraints
| Parameter | Value | Source |
|-----------|-------|--------|
| Nominal Voltage | 230V | UK Standard |
| Tolerance | ±10% | UK G59/3 |
| Minimum | 207V | Calculated |
| Maximum | 253V | Calculated |

### Power Factor Constraints
| Parameter | Value | Source |
|-----------|-------|--------|
| Minimum | 0.8 | Typical |
| Maximum | 1.0 | Physical limit |
| Recommended | 0.95-1.0 | Best practice |

### Capacity Constraints
Transformer ratings vary by feeder (from SSEN metadata):
- Typical residential: 50-500 kVA
- Large feeders: 500-1000 kVA
- Peak load should not exceed 80% of rated capacity under normal conditions

### Household-Level Constraints
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max instantaneous power | 100 kW | Fuse rating + safety margin |
| Typical max | 10-15 kW | UK domestic supply |
| Min consumption | 0 kWh | Physical limit |
| Max zero duration | 48 hours | Occupied household baseline |

## SSEN Metadata Processing

The SSEN dataset provides **metadata only** (no time-series consumption data):
- 416,609 feeder records
- Voltage levels
- Transformer capacities
- Geographic locations
- Postcode sectors

This metadata is used to:
1. Define physical constraints for verifier
2. Validate anomaly candidates against real network limits
3. Inform realistic scenario generation

## No Ground Truth Strategy

Since we have no labeled anomalies, we employ:

1. **Synthetic injection for quantitative metrics**
   - Controlled anomaly types
   - Known ground truth
   - Reproducible evaluation

2. **Physics-based validation for real data**
   - SSEN constraint violations
   - Domain knowledge rules
   - Expert review of flagged cases

3. **Comparative evaluation**
   - Multiple baseline methods
   - Ensemble agreement
   - Cross-validation across datasets

4. **Self-play as implicit labeling**
   - Verifier learns from proposer adversarial examples
   - No manual labels needed
   - Continuous improvement through iteration

## References

- UK G59/3: Voltage limits for distribution networks
- SSEN Technical Standards: Transformer loading limits
- Statistical anomaly detection: Hodge & Austin (2004)
- Self-play learning: Silver et al. (2017) - AlphaGo Zero
- Isolation Forest: Liu et al. (2008)
- UK Electrical Safety Standards BS 7671:2018

---
*Last updated: 2025-10-18*  
*Author: Vatsal Mehta*  
*Project: FYP - Predictive Anomaly Detection using Self-Play RL*

