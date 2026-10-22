# Sensor Data Homogenization Workflow

## Quick Start

### Step 1: Run Data Preparation
```bash
python data_prep.py
```

This script performs **two operations**:
1. **Unification**: Merges all three CSV files by timestamp into `unified_sensor_data.csv`
2. **Homogenization**: Bins data into regular 5-minute intervals and forward-fills values into `homogenized_sensor_data.csv`

### Step 2: Analyze Results (Optional)
```bash
python analyze_homogenization.py
```

Shows before/after statistics and data quality metrics.

---

## What Gets Created

| File | Purpose | Timestamps | Rows | Use Case |
|------|---------|-----------|------|----------|
| `unified_sensor_data.csv` | All sensor data merged by timestamp | Irregular | ~13,145 | Data inspection, raw merged view |
| `homogenized_sensor_data.csv` | Regular time series with forward-filled values | Regular 5-min | ~1,800+ | Time-series analysis, ML models, visualization |

---

## Data Transformation Example

### Before (3 separate irregular files):
```
EC data         PH/ORP data         Temperature data
21:32:25 ✓                          21:32:06 ✓
21:35:00 ✓                          21:32:25 ✓
21:40:00 ✓      21:44:33 ✓          21:33:45 ✓
21:45:00 ✓      21:44:33 ✓          21:34:25 ✓
...                                 ...
```

### After (Single homogenized file with 5-min bins):
```
Time             EC    pH   Temp1  Temp2  Temp3  ...
21:30:00  (ffill)
21:35:00  ✓      ✓     ✗    ✓      ✓      ✓
21:40:00  ✓      ✗    ✓      ✓      ✓      ✓  (forward-filled from 21:35)
21:45:00  ✓      ✗    ✓      ✓      ✓      ✓  (forward-filled from 21:40)
21:50:00  ✓      ✗    ✓      ✓      ✓      ✓  (forward-filled from 21:45)
```

---

## Understanding Forward-Fill

**Forward-fill** means: if a sensor didn't report in this 5-minute window, use the last known value.

✅ **Assumption**: Sensor values change slowly (temperature, pH, conductivity)

🔴 **Not suitable for**: Rapid events, discrete measurements, alert thresholds

---

## Customization

### Change Time Interval
Edit `data_prep.py` line 57:
```python
interval = "5min"   # Change to "10min", "15min", etc.
```

### Change Aggregation Strategy
Edit `data_prep.py` line 67:
```python
# Current: takes first measurement in each window
df_homogenized = df_numeric.set_index('Time').resample(interval).agg({
    col: 'first' for col in df_numeric.columns if col != 'Time'
})

# Alternative: average all measurements
df_homogenized = df_numeric.set_index('Time').resample(interval).mean()

# Alternative: linear interpolation instead of forward-fill
df_homogenized = df_numeric.set_index('Time').resample(interval).interpolate(method='linear')
```

---

## Files Included

- **`data_prep.py`** - Main script (unify + homogenize)
- **`analyze_homogenization.py`** - Statistics and comparison
- **`DATA_HOMOGENIZATION_GUIDE.md`** - Detailed documentation
- **`README_HOMOGENIZATION.md`** - This file

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"
```bash
pip install pandas numpy
```

### "No such file or directory" for data files
Make sure CSV files are in the `data/` subdirectory:
```
data/
├── EC-data-as-joinbyfield-2026-10-21 11_34_59.csv
├── PH _ ORP-data-as-joinbyfield-2026-10-21 11_34_20.csv
└── Temperatur-data-as-joinbyfield-2026-10-21 11_35_25.csv
```

---

## Next Steps

1. ✅ Run `python data_prep.py`
2. ✅ Inspect `homogenized_sensor_data.csv`
3. ✅ Use for analysis, visualization, or machine learning
4. ✅ (Optional) Run `python analyze_homogenization.py` for metrics

---

## Questions?

See `DATA_HOMOGENIZATION_GUIDE.md` for:
- Why homogenization improves data
- Alternative strategies
- When to use each output format
- Use cases and limitations
