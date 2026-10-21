"""Unify aquaponics sensor data from multiple CSV files."""

import pandas as pd
from pathlib import Path

# Define file paths
data_dir = Path(__file__).parent / "data"
ec_file = data_dir / "EC-data-as-joinbyfield-2026-10-21 11_34_59.csv"
ph_orp_file = data_dir / "PH _ ORP-data-as-joinbyfield-2026-10-21 11_34_20.csv"
temp_file = data_dir / "Temperatur-data-as-joinbyfield-2026-10-21 11_35_25.csv"

# Read CSV files
print("Reading sensor data files...")
df_ec = pd.read_csv(ec_file)
df_ph_orp = pd.read_csv(ph_orp_file)
df_temp = pd.read_csv(temp_file)

print(f"EC data shape: {df_ec.shape}")
print(f"PH/ORP data shape: {df_ph_orp.shape}")
print(f"Temperature data shape: {df_temp.shape}")

# Convert Time to datetime for proper merging
df_ec["Time"] = pd.to_datetime(df_ec["Time"])
df_ph_orp["Time"] = pd.to_datetime(df_ph_orp["Time"])
df_temp["Time"] = pd.to_datetime(df_temp["Time"])

# Merge all dataframes on Time using outer join to preserve all timestamps
print("\nMerging datasets...")
df_unified = df_ec.merge(df_ph_orp, on="Time", how="outer", suffixes=("_ec", "_ph"))
df_unified = df_unified.merge(df_temp, on="Time", how="outer", suffixes=("", "_temp"))

# Sort by time
df_unified = df_unified.sort_values("Time").reset_index(drop=True)

print(f"Unified data shape: {df_unified.shape}")
print(f"\nColumn names:")
for i, col in enumerate(df_unified.columns, 1):
    print(f"  {i}. {col}")

# Handle duplicate temperature columns by keeping the most reliable one
# Remove redundant columns with same data
if "temp_sensor_93982118_C_ec" in df_unified.columns and "temp_sensor_93982118_C" in df_unified.columns:
    # Keep the first one and drop duplicates
    df_unified = df_unified.drop(columns=[col for col in df_unified.columns 
                                          if col.endswith("_ec") and col in df_unified.columns 
                                          and col.replace("_ec", "") in df_unified.columns])

# Clean up column names - standardize formatting
df_unified.columns = [col.replace(" ", "_").strip() for col in df_unified.columns]

# Save unified dataset
output_file = data_dir / "unified_sensor_data.csv"
df_unified.to_csv(output_file, index=False)
print(f"\n✓ Unified data saved to: {output_file}")

# Display summary statistics
print(f"\nSummary Statistics:")
print(f"  Time range: {df_unified['Time'].min()} to {df_unified['Time'].max()}")
print(f"  Total records: {len(df_unified)}")
print(f"  Duration: {df_unified['Time'].max() - df_unified['Time'].min()}")

# Show data quality
print(f"\nData quality (non-null counts):")
for col in df_unified.columns:
    if col != "Time":
        non_null = df_unified[col].notna().sum()
        pct = (non_null / len(df_unified)) * 100
        print(f"  {col}: {non_null} / {len(df_unified)} ({pct:.1f}%)")