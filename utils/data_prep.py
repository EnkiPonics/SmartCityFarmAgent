"""Unify and homogenize aquaponics sensor data from multiple CSV files."""

import pandas as pd
import numpy as np
import re
from pathlib import Path

def extract_numeric_value(val):
    """Extract numeric value from strings like '1053 uS/cm', '7.83 ph', '17.3 °C'."""
    if pd.isna(val) or val == '':
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        match = re.search(r'[-+]?\d*\.?\d+', val.replace(',', '.'))
        if match:
            return float(match.group())
    return np.nan

# ============================================================================
# STEP 1: Unify data from multiple CSV files
# ============================================================================

# Define file paths
data_dir = Path(__file__).parent / "data"
ec_file = data_dir / "EC-data-as-joinbyfield-2026-10-21 11_34_59.csv"
ph_orp_file = data_dir / "PH _ ORP-data-as-joinbyfield-2026-10-21 11_34_20.csv"
temp_file = data_dir / "Temperatur-data-as-joinbyfield-2026-10-21 11_35_25.csv"

print("="*70)
print("STEP 1: Unifying sensor data from multiple CSV files")
print("="*70)

# Read CSV files
print("\nReading sensor data files...")
df_ec = pd.read_csv(ec_file)
df_ph_orp = pd.read_csv(ph_orp_file)
df_temp = pd.read_csv(temp_file)

print(f"  EC data: {df_ec.shape[0]} rows")
print(f"  PH/ORP data: {df_ph_orp.shape[0]} rows")
print(f"  Temperature data: {df_temp.shape[0]} rows")

# Convert Time to datetime for proper merging
df_ec["Time"] = pd.to_datetime(df_ec["Time"])
df_ph_orp["Time"] = pd.to_datetime(df_ph_orp["Time"])
df_temp["Time"] = pd.to_datetime(df_temp["Time"])

# Merge all dataframes on Time using outer join
print("\nMerging datasets by timestamp...")
df_unified = df_ec.merge(df_ph_orp, on="Time", how="outer", suffixes=("_ec", "_ph"))
df_unified = df_unified.merge(df_temp, on="Time", how="outer", suffixes=("", "_temp"))

# Sort by time
df_unified = df_unified.sort_values("Time").reset_index(drop=True)

print(f"Unified data shape: {df_unified.shape[0]} rows, {df_unified.shape[1]} columns")

# Clean up duplicate columns
if "temp_sensor_93982118_C_ec" in df_unified.columns and "temp_sensor_93982118_C" in df_unified.columns:
    df_unified = df_unified.drop(columns=[col for col in df_unified.columns 
                                          if col.endswith("_ec") and col in df_unified.columns 
                                          and col.replace("_ec", "") in df_unified.columns])

# Standardize column names
df_unified.columns = [col.replace(" ", "_").strip() for col in df_unified.columns]

# Save unified dataset
unified_file = data_dir / "unified_sensor_data.csv"
df_unified.to_csv(unified_file, index=False)
print(f"✓ Unified data saved to: {unified_file.name}\n")

# ============================================================================
# STEP 2: Homogenize data - bin into regular intervals and forward-fill
# ============================================================================

print("="*70)
print("STEP 2: Homogenizing data with 5-minute binning and forward-filling")
print("="*70)

print("\nConverting all values to numeric (removing units)...")
df_numeric = df_unified.copy()
for col in df_numeric.columns:
    if col != 'Time':
        df_numeric[col] = df_numeric[col].apply(extract_numeric_value)

# Create regular 5-minute intervals
print("Creating 5-minute time bins...")
interval = "5min"
df_homogenized = df_numeric.set_index('Time').resample(interval).agg({
    col: 'first' for col in df_numeric.columns if col != 'Time'
})

# Forward-fill: carry last known value forward
# This assumes sensor values don't change until next measurement
print("Forward-filling missing values (carrying last measurement forward)...")
df_homogenized = df_homogenized.ffill()

# Backward-fill any remaining NaNs at the start
df_homogenized = df_homogenized.bfill()

# Reset index to bring Time back as a column
df_homogenized = df_homogenized.reset_index()

print(f"\nHomogenized data shape: {df_homogenized.shape[0]} rows, {df_homogenized.shape[1]} columns")
print(f"Time range: {df_homogenized['Time'].min()} to {df_homogenized['Time'].max()}")

# Save homogenized dataset
homogenized_file = data_dir / "homogenized_sensor_data.csv"
df_homogenized.to_csv(homogenized_file, index=False)
print(f"✓ Homogenized data saved to: {homogenized_file.name}\n")

# ============================================================================
# REPORTING
# ============================================================================

print("="*70)
print("Data Quality Report - Homogenized Dataset")
print("="*70)

print(f"\nBefore homogenization (Unified): {len(df_unified)} rows with irregular timestamps")
print(f"After homogenization: {len(df_homogenized)} rows with regular 5-minute intervals")
print(f"Coverage improvement: {(len(df_homogenized) / len(df_unified) * 100):.1f}% of timeline filled")

print(f"\nSensor data completeness (after homogenization):")
for col in df_homogenized.columns:
    if col != 'Time':
        non_null = df_homogenized[col].notna().sum()
        pct = (non_null / len(df_homogenized)) * 100
        print(f"  {col}: {non_null:5d} / {len(df_homogenized)} ({pct:5.1f}%)")

print("\n" + "="*70)
print("✓ Data preparation complete!")
print("="*70)