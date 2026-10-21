"""Compare before/after data homogenization and generate statistics."""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_data_quality(df, name):
    """Analyze completeness and statistics for a dataset."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    print(f"\nDimensions: {df.shape[0]} rows × {df.shape[1]} columns")
    
    if 'Time' in df.columns:
        time_range = pd.to_datetime(df['Time'].min()) if isinstance(df['Time'].iloc[0], str) else df['Time'].min()
        time_max = pd.to_datetime(df['Time'].max()) if isinstance(df['Time'].iloc[0], str) else df['Time'].max()
        duration = time_max - time_range
        print(f"Time range: {time_range} to {time_max}")
        print(f"Duration: {duration}")
        print(f"Avg time between readings: {duration / len(df)}")
    
    print(f"\nData completeness by column:")
    completeness = {}
    for col in df.columns:
        if col != 'Time':
            non_null = df[col].notna().sum()
            pct = (non_null / len(df)) * 100
            completeness[col] = pct
            status = "✓" if pct > 80 else "◐" if pct > 50 else "✗"
            print(f"  {status} {col}: {non_null:6d} / {len(df)} ({pct:5.1f}%)")
    
    avg_completeness = np.mean(list(completeness.values()))
    print(f"\nAverage completeness: {avg_completeness:.1f}%")
    
    return completeness

def main():
    data_dir = Path(__file__).parent / "data"
    unified_file = data_dir / "unified_sensor_data.csv"
    homogenized_file = data_dir / "homogenized_sensor_data.csv"
    
    # Load data
    print("Loading datasets...")
    if unified_file.exists():
        df_unified = pd.read_csv(unified_file)
        analyze_data_quality(df_unified, "BEFORE: Unified Data (Irregular Timestamps)")
    else:
        print(f"⚠ {unified_file.name} not found. Run data_prep.py first.")
        return
    
    if homogenized_file.exists():
        df_homogenized = pd.read_csv(homogenized_file)
        analyze_data_quality(df_homogenized, "AFTER: Homogenized Data (5-min Regular Intervals)")
        
        # Comparison metrics
        print(f"\n{'='*70}")
        print("Comparison: Before vs After Homogenization")
        print(f"{'='*70}")
        
        row_reduction = (1 - len(df_homogenized) / len(df_unified)) * 100
        print(f"\nRow compression: {len(df_unified):5d} → {len(df_homogenized):5d} rows ({row_reduction:5.1f}% reduction)")
        print(f"Regularization: Irregular timestamps → Fixed 5-minute intervals")
        
        # Show sample transformation
        print(f"\n{'='*70}")
        print("Sample Transformation Example")
        print(f"{'='*70}")
        
        # Get first 20 unified rows
        sample_unified = df_unified.head(20).copy()
        print("\nBefore (Unified - first 10 rows with irregular timestamps):")
        print(sample_unified[['Time'] + [col for col in sample_unified.columns if col != 'Time'][:5]].to_string(index=False))
        
        # Get corresponding homogenized rows
        if 'Time' in df_homogenized.columns:
            first_time = pd.to_datetime(df_homogenized['Time'].iloc[0])
            tenth_time = first_time + pd.Timedelta(minutes=45)
            sample_homogenized = df_homogenized[
                (pd.to_datetime(df_homogenized['Time']) >= first_time) &
                (pd.to_datetime(df_homogenized['Time']) <= tenth_time)
            ].copy()
            
            print(f"\n\nAfter (Homogenized - same time period with 5-min intervals):")
            print(sample_homogenized[['Time'] + [col for col in sample_homogenized.columns if col != 'Time'][:5]].to_string(index=False))
            
            print(f"\nNote: Each homogenized row represents a 5-minute time window with forward-filled values")
            print(f"      where measurements are missing. This creates a continuous time series suitable")
            print(f"      for time-series analysis, machine learning, and visualization.")
    else:
        print(f"⚠ {homogenized_file.name} not found. Run data_prep.py first.")

if __name__ == "__main__":
    main()
