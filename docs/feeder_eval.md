# Feeder-Level Evaluation Methodology

This document outlines the comprehensive evaluation framework for validating pseudo-feeder aggregations against real SSEN Low Voltage (LV) feeder data, ensuring our household-level models produce realistic distribution network characteristics.

## Evaluation Overview

### Objective
Validate that aggregated household forecasts (from UK-DALE and LCL data) exhibit realistic distribution network characteristics when compared to actual SSEN LV feeder measurements.

### Key Evaluation Dimensions
1. **Statistical Distributions**: Load magnitude, variability, and temporal patterns
2. **Peak Load Analysis**: Timing, magnitude, and duration of peak events  
3. **Ramp Rate Characteristics**: Rate of change limitations and typical patterns
4. **Seasonal Patterns**: Long-term consumption variations and weather sensitivity
5. **Anomaly Case Studies**: Unusual events and their network-level manifestations

## Pseudo-Feeder Construction

### Aggregation Methodology
```python
class PseudoFeederBuilder:
    """Construct realistic distribution feeders from household data."""
    
    def __init__(self, household_forecasts, diversity_factors, transformer_capacity):
        self.households = household_forecasts
        self.diversity_factors = diversity_factors  # Based on SSEN statistics
        self.capacity = transformer_capacity        # Typical 200-500 kVA
        
    def construct_feeder(self, num_households=50):
        """Aggregate households with realistic diversity effects."""
        
        # 1. Select diverse household types
        household_mix = self.select_representative_mix(num_households)
        
        # 2. Apply diversity factors (not simple summation)
        aggregated_load = self.apply_diversity_factors(household_mix)
        
        # 3. Add network losses and transformer characteristics
        feeder_load = self.add_network_effects(aggregated_load)
        
        return feeder_load
    
    def select_representative_mix(self, n_households):
        """Select households representing typical feeder demographics."""
        household_types = {
            'high_consumption': 0.15,    # Large houses, high income
            'medium_consumption': 0.60,  # Typical households  
            'low_consumption': 0.20,     # Small houses, efficient
            'ev_owners': 0.25,           # Electric vehicle ownership
            'heat_pump': 0.10            # Heat pump installations
        }
        return self.sample_by_type(household_types, n_households)
```

### Diversity Factor Application
Real distribution feeders exhibit diversity effects where peak loads don't align perfectly across households:

```python
class DiversityFactorModel:
    """Model realistic load diversity in distribution feeders."""
    
    def __init__(self, ssen_calibration_data):
        # Calibrated from SSEN feeder observations
        self.peak_diversity_factors = {
            10: 0.85,   # 10 households: 85% of sum of individual peaks
            25: 0.70,   # 25 households: 70% of sum  
            50: 0.60,   # 50 households: 60% of sum
            100: 0.55   # 100 households: 55% of sum
        }
        
    def apply_temporal_diversity(self, household_loads):
        """Apply time-varying diversity effects."""
        n_households = len(household_loads)
        diversity_factor = self.interpolate_diversity_factor(n_households)
        
        # Time-varying diversity (peaks less coincident)
        time_diversity = self.compute_time_varying_diversity(household_loads)
        
        return household_loads.sum(axis=0) * diversity_factor * time_diversity
```

## Statistical Distribution Comparison

### Core Statistical Tests

#### 1. Kolmogorov-Smirnov Test
```python
def distributional_comparison(pseudo_feeder_loads, ssen_feeder_loads):
    """Compare load distributions using KS test."""
    
    results = {}
    
    # Overall load distribution
    ks_stat, p_value = kstest(pseudo_feeder_loads, ssen_feeder_loads)
    results['overall_distribution'] = {
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'significant_difference': p_value < 0.05
    }
    
    # Time-of-day distributions
    for hour in range(24):
        pseudo_hour = pseudo_feeder_loads[pseudo_feeder_loads.index.hour == hour]
        ssen_hour = ssen_feeder_loads[ssen_feeder_loads.index.hour == hour]
        
        ks_stat_hour, p_val_hour = kstest(pseudo_hour, ssen_hour)
        results[f'hour_{hour:02d}'] = {
            'ks_statistic': ks_stat_hour,
            'p_value': p_val_hour
        }
    
    return results
```

#### 2. Wasserstein Distance
```python
def wasserstein_analysis(pseudo_loads, ssen_loads):
    """Compute Earth Mover's Distance between load distributions."""
    
    # Overall distribution distance
    overall_distance = wasserstein_distance(pseudo_loads, ssen_loads)
    
    # Seasonal distribution distances
    seasonal_distances = {}
    for season in ['spring', 'summer', 'autumn', 'winter']:
        pseudo_seasonal = extract_seasonal_data(pseudo_loads, season)
        ssen_seasonal = extract_seasonal_data(ssen_loads, season)
        
        seasonal_distances[season] = wasserstein_distance(
            pseudo_seasonal, ssen_seasonal
        )
    
    return {
        'overall_distance': overall_distance,
        'seasonal_distances': seasonal_distances,
        'normalized_distance': overall_distance / np.mean(ssen_loads)
    }
```

### Load Duration Curve Analysis
```python
def load_duration_curve_comparison(pseudo_loads, ssen_loads):
    """Compare load duration characteristics."""
    
    # Sort loads in descending order (duration curve)
    pseudo_sorted = np.sort(pseudo_loads)[::-1]
    ssen_sorted = np.sort(ssen_loads)[::-1]
    
    # Key percentiles for comparison
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    
    comparison = {}
    for p in percentiles:
        pseudo_val = np.percentile(pseudo_sorted, 100-p)
        ssen_val = np.percentile(ssen_sorted, 100-p)
        
        comparison[f'p{p}'] = {
            'pseudo_feeder': pseudo_val,
            'ssen_feeder': ssen_val,
            'relative_error': abs(pseudo_val - ssen_val) / ssen_val
        }
    
    return comparison
```

## Peak Load Analysis

### Peak Event Detection
```python
class PeakAnalyzer:
    """Analyze peak load events and their characteristics."""
    
    def __init__(self, threshold_percentile=95):
        self.threshold_percentile = threshold_percentile
        
    def identify_peak_events(self, load_data):
        """Identify peak load events above threshold."""
        threshold = np.percentile(load_data, self.threshold_percentile)
        
        # Find periods above threshold
        above_threshold = load_data > threshold
        
        # Group consecutive periods into events
        peak_events = []
        in_event = False
        event_start = None
        
        for i, is_peak in enumerate(above_threshold):
            if is_peak and not in_event:
                # Start of new peak event
                in_event = True
                event_start = i
            elif not is_peak and in_event:
                # End of peak event
                in_event = False
                peak_events.append({
                    'start': event_start,
                    'end': i-1,
                    'duration': i - event_start,
                    'peak_magnitude': load_data[event_start:i].max(),
                    'average_magnitude': load_data[event_start:i].mean()
                })
        
        return peak_events
        
    def compare_peak_characteristics(self, pseudo_events, ssen_events):
        """Compare peak event characteristics between datasets."""
        
        comparison = {}
        
        # Peak frequency
        comparison['event_frequency'] = {
            'pseudo_feeder': len(pseudo_events),
            'ssen_feeder': len(ssen_events),
            'frequency_ratio': len(pseudo_events) / len(ssen_events)
        }
        
        # Peak duration distribution  
        pseudo_durations = [event['duration'] for event in pseudo_events]
        ssen_durations = [event['duration'] for event in ssen_events]
        
        comparison['duration_statistics'] = {
            'pseudo_mean': np.mean(pseudo_durations),
            'ssen_mean': np.mean(ssen_durations),
            'pseudo_std': np.std(pseudo_durations),
            'ssen_std': np.std(ssen_durations)
        }
        
        # Peak magnitude distribution
        pseudo_magnitudes = [event['peak_magnitude'] for event in pseudo_events]
        ssen_magnitudes = [event['peak_magnitude'] for event in ssen_events]
        
        comparison['magnitude_statistics'] = {
            'pseudo_mean': np.mean(pseudo_magnitudes),
            'ssen_mean': np.mean(ssen_magnitudes),
            'magnitude_correlation': np.corrcoef(
                pseudo_magnitudes[:min(len(pseudo_magnitudes), len(ssen_magnitudes))],
                ssen_magnitudes[:min(len(pseudo_magnitudes), len(ssen_magnitudes))]
            )[0,1]
        }
        
        return comparison
```

### Peak Timing Analysis
```python
def peak_timing_analysis(pseudo_loads, ssen_loads):
    """Analyze temporal patterns of peak loads."""
    
    # Daily peak timing
    pseudo_daily_peaks = pseudo_loads.groupby(pseudo_loads.index.date).idxmax()
    ssen_daily_peaks = ssen_loads.groupby(ssen_loads.index.date).idxmax()
    
    # Extract hour of peak
    pseudo_peak_hours = [peak.hour for peak in pseudo_daily_peaks]
    ssen_peak_hours = [peak.hour for peak in ssen_daily_peaks]
    
    # Peak hour distribution comparison
    peak_timing_comparison = {
        'pseudo_peak_hour_distribution': np.bincount(pseudo_peak_hours, minlength=24),
        'ssen_peak_hour_distribution': np.bincount(ssen_peak_hours, minlength=24),
        'peak_hour_correlation': np.corrcoef(
            np.bincount(pseudo_peak_hours, minlength=24),
            np.bincount(ssen_peak_hours, minlength=24)
        )[0,1]
    }
    
    return peak_timing_comparison
```

## Ramp Rate Analysis

### Rate of Change Characteristics
```python
class RampRateAnalyzer:
    """Analyze load change rates and ramping characteristics."""
    
    def compute_ramp_rates(self, load_data, time_delta_minutes=30):
        """Compute load change rates."""
        # Convert to rate per hour
        ramp_rates = load_data.diff() / (time_delta_minutes / 60)
        return ramp_rates.dropna()
    
    def analyze_ramp_characteristics(self, pseudo_loads, ssen_loads):
        """Compare ramping behavior between pseudo and real feeders."""
        
        pseudo_ramps = self.compute_ramp_rates(pseudo_loads)
        ssen_ramps = self.compute_ramp_rates(ssen_loads)
        
        analysis = {
            # Basic statistics
            'ramp_rate_statistics': {
                'pseudo_mean_up': pseudo_ramps[pseudo_ramps > 0].mean(),
                'pseudo_mean_down': pseudo_ramps[pseudo_ramps < 0].mean(),
                'ssen_mean_up': ssen_ramps[ssen_ramps > 0].mean(),
                'ssen_mean_down': ssen_ramps[ssen_ramps < 0].mean(),
                'pseudo_std': pseudo_ramps.std(),
                'ssen_std': ssen_ramps.std()
            },
            
            # Extreme ramp events
            'extreme_ramps': {
                'pseudo_99th_percentile': np.percentile(np.abs(pseudo_ramps), 99),
                'ssen_99th_percentile': np.percentile(np.abs(ssen_ramps), 99),
                'pseudo_max_up': pseudo_ramps.max(),
                'pseudo_max_down': pseudo_ramps.min(),
                'ssen_max_up': ssen_ramps.max(),
                'ssen_max_down': ssen_ramps.min()
            },
            
            # Ramp rate distribution comparison
            'distribution_similarity': {
                'ks_test': kstest(pseudo_ramps, ssen_ramps),
                'wasserstein_distance': wasserstein_distance(pseudo_ramps, ssen_ramps)
            }
        }
        
        return analysis
```

## Anomaly Case Studies

### Anomaly Detection Framework
```python
class FeederAnomalyAnalyzer:
    """Detect and analyze anomalies at feeder level."""
    
    def __init__(self, detection_methods=['isolation_forest', 'lof', 'statistical']):
        self.detection_methods = detection_methods
        
    def detect_anomalies(self, feeder_loads):
        """Multi-method anomaly detection."""
        
        anomalies = {}
        
        # Isolation Forest
        if 'isolation_forest' in self.detection_methods:
            iso_forest = IsolationForest(contamination=0.05)
            iso_anomalies = iso_forest.fit_predict(feeder_loads.values.reshape(-1, 1))
            anomalies['isolation_forest'] = iso_anomalies == -1
            
        # Local Outlier Factor
        if 'lof' in self.detection_methods:
            lof = LocalOutlierFactor(contamination=0.05)
            lof_anomalies = lof.fit_predict(feeder_loads.values.reshape(-1, 1))
            anomalies['lof'] = lof_anomalies == -1
            
        # Statistical outliers (3-sigma rule)
        if 'statistical' in self.detection_methods:
            z_scores = np.abs((feeder_loads - feeder_loads.mean()) / feeder_loads.std())
            anomalies['statistical'] = z_scores > 3
            
        return anomalies
    
    def compare_anomaly_patterns(self, pseudo_anomalies, ssen_anomalies):
        """Compare anomaly detection results between datasets."""
        
        comparison = {}
        
        for method in self.detection_methods:
            pseudo_anom = pseudo_anomalies[method]
            ssen_anom = ssen_anomalies[method]
            
            comparison[method] = {
                'pseudo_anomaly_rate': np.mean(pseudo_anom),
                'ssen_anomaly_rate': np.mean(ssen_anom),
                'temporal_correlation': self.compute_temporal_correlation(
                    pseudo_anom, ssen_anom
                ),
                'pattern_similarity': self.compute_pattern_similarity(
                    pseudo_anom, ssen_anom
                )
            }
            
        return comparison
```

### Case Study Templates

#### Case Study 1: Winter Peak Events
```markdown
## Winter Peak Event Analysis

### Event Description
- **Date Range**: [Specific winter period]
- **Weather Context**: [Temperature, wind, precipitation data]
- **Expected Pattern**: [Normal winter consumption baseline]

### Pseudo-Feeder Response
- **Peak Magnitude**: [Maximum load during event]
- **Duration**: [Hours above baseline + X%]
- **Ramp Characteristics**: [Rate of increase/decrease]
- **Household Contributions**: [Which household types drove the peak]

### SSEN Feeder Comparison
- **Actual Peak Magnitude**: [Real feeder measurement]
- **Magnitude Error**: [Percentage difference]
- **Timing Accuracy**: [Hours difference in peak timing]
- **Pattern Similarity**: [Correlation coefficient]

### Analysis
- **Realism Assessment**: [Qualitative evaluation]
- **Model Performance**: [Quantitative metrics]
- **Lessons Learned**: [Insights for model improvement]
```

#### Case Study 2: Equipment Failure Events
```markdown
## Equipment Failure Simulation

### Scenario Setup
- **Failure Type**: [Heating system, EV charger, major appliance]
- **Affected Households**: [Number and characteristics]
- **Failure Duration**: [Hours of reduced consumption]

### Aggregated Impact
- **Load Reduction**: [Magnitude at feeder level]
- **Recovery Pattern**: [How consumption returned to normal]
- **Diversity Effects**: [How individual failures aggregated]

### Validation Against Real Events
- **Historical Precedent**: [Similar real events in SSEN data]
- **Magnitude Comparison**: [Predicted vs. actual impact]
- **Recovery Realism**: [Pattern matching analysis]
```

## Evaluation Report Template

### Executive Summary
- **Overall Realism Score**: [Composite metric 0-100]
- **Key Strengths**: [Where pseudo-feeders match SSEN well]
- **Areas for Improvement**: [Where discrepancies exist]
- **Recommended Model Updates**: [Specific improvements needed]

### Detailed Metrics
```python
evaluation_results = {
    'statistical_comparison': {
        'ks_test_p_value': 0.23,           # >0.05 indicates good match
        'wasserstein_distance': 12.4,       # Lower is better
        'load_duration_curve_error': 0.08   # Mean absolute percentage error
    },
    
    'peak_analysis': {
        'peak_timing_correlation': 0.87,     # Daily peak hour correlation
        'peak_magnitude_error': 0.12,       # MAPE on peak magnitudes
        'peak_frequency_ratio': 1.05        # Pseudo/SSEN peak event ratio
    },
    
    'ramp_rate_analysis': {
        'ramp_distribution_similarity': 0.91, # Correlation coefficient
        'extreme_ramp_accuracy': 0.78,        # 99th percentile accuracy
        'average_ramp_error': 0.15            # MAPE on average ramp rates
    },
    
    'anomaly_detection': {
        'anomaly_pattern_correlation': 0.72,  # Temporal anomaly correlation
        'false_positive_rate': 0.08,          # Anomalies in pseudo not in SSEN
        'false_negative_rate': 0.12           # Anomalies in SSEN not in pseudo
    }
}
```

### Visualizations
All evaluation reports include standardized figures saved in `docs/figures/`:

1. **`load_duration_curves.png`**: Pseudo vs SSEN load duration comparison
2. **`peak_timing_distribution.png`**: Daily peak hour distributions
3. **`ramp_rate_distributions.png`**: Load change rate histograms
4. **`seasonal_patterns.png`**: Month-by-month consumption patterns
5. **`anomaly_timeline.png`**: Detected anomalies overlaid on load data
6. **`statistical_qq_plots.png`**: Quantile-quantile plots for distribution comparison

### Recommendations
Based on evaluation results, provide specific recommendations for:
- Model architecture improvements
- Training data augmentation needs
- Constraint refinement in verifier component
- Additional validation requirements

This comprehensive evaluation framework ensures that our household-level forecasting models produce realistic distribution network characteristics, validating the practical applicability of our self-play approach to real-world energy systems.
