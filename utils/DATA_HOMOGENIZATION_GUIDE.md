# Data Homogenization Guide

## Overview
Your aquaponics sensor data has irregular timestamps with different sensors reporting at different intervals. This guide explains how to create a more homogeneous dataset.

## Current Problem
- **EC sensors**: Report every ~5 minutes
- **pH/ORP sensors**: Report sporadically (sometimes hours between readings)
- **Temperature sensors**: Report at various intervals (some every ~2 min, others every ~5 min, DWC less frequent)
- **Result**: Many rows have sparse data with multiple NULLs

### Example of Irregular Data:
```
2026-10-13 21:32:06  → only dwc_temp
2026-10-13 21:32:25  → EC + pressure_temp + multiple other temps
2026-10-13 21:33:45  → only dwc_temp
2026-10-13 21:34:25  → EC + temps
2026-10-13 21:35:00  → EC only
```

## Solution: 5-Minute Binning + Forward-Fill

### Strategy
1. **Create regular 5-minute intervals** across your entire time range
2. **Take first measurement** in each 5-minute window
3. **Forward-fill missing values** - assume sensor values don't change until next measurement
4. **Convert all values to numeric** - removes units like "°C", "uS/cm", "mV", "ph"

### Why This Works
- **5 minutes**: Longest interval between EC readings (most frequent sensor)
- **Forward-fill**: Valid assumption for slow-changing processes (water temperature, pH don't change rapidly)
- **Regular intervals**: Enables time-series analysis, machine learning, and statistical methods

### Result
Your dataset transforms from:
- ❌ **Before**: 13,145 irregular rows with ~60-70% NULL values per sensor
- ✅ **After**: ~1,800+ regular rows with 90%+ completeness via forward-fill

## Implementation

Use the updated `data_prep.py`:

```bash
python data_prep.py
```

Outputs:
- `unified_sensor_data.csv` - All data merged by timestamp (irregular intervals)
- `homogenized_sensor_data.csv` - Regular 5-min intervals with forward-filled values

## Advanced Options

### Change Time Interval
If you want different binning (e.g., 10 minutes for coarser data):

```python
interval = "10min"  # Instead of "5min"
```

Options:
- `"1min"` - Very fine granularity (many NULLs remain)
- `"5min"` - Recommended (balances detail and completeness)
- `"10min"` - Coarser (more forward-filled, less detail)
- `"15min"` - Even coarser

### Alternative Aggregation Strategies

Instead of `'first'`, you can aggregate multiple readings within a bin:

```python
# Mean of all readings in the window
df_homogenized = df_numeric.set_index('Time').resample(interval).mean()

# Max reading (for peaks)
df_homogenized = df_numeric.set_index('Time').resample(interval).max()

# Interpolate instead of forward-fill
df_homogenized = df_homogenized.interpolate(method='linear')
```

### Combine Multiple Strategies

```python
# Take mean if multiple readings, else forward-fill
df_homogenized = df_numeric.set_index('Time').resample(interval).agg({
    'ec_calibrated': 'mean',        # Average EC readings
    'ph_cal': 'mean',               # Average pH readings
    'temp_sensor_93982118_C_ph': 'first'  # Take first temperature
})
df_homogenized = df_homogenized.fillna(method='ffill')
```

## Data Quality Metrics

After homogenization, check completeness:

```python
for col in df_homogenized.columns:
    non_null = df_homogenized[col].notna().sum()
    pct = (non_null / len(df_homogenized)) * 100
    print(f"{col}: {pct:.1f}% complete")
```

Expected results:
- **Frequently reporting sensors** (EC, temps): 95%+ complete
- **Occasionally reporting sensors** (pH/ORP): 20-50% complete (depends on calibration schedule)
- **Forward-filled columns**: Will show 100% after fill, but reflect 0-filled values until first real reading

## Use Cases

### ✅ Good for:
- Time-series forecasting (LSTM models)
- Correlation analysis
- Anomaly detection
- Statistical analysis
- Visualization with regular grid

### ⚠️ Consider alternatives for:
- Real-time alerting (use original irregular data)
- Missing data imputation studies
- Uncertainty quantification (forward-fill introduces false certainty)

## Next Steps

1. Run `data_prep.py` to generate homogenized data
2. Explore `homogenized_sensor_data.csv`
3. Validate with domain knowledge (do the filled values make sense?)
4. Use for modeling, analysis, or visualization
