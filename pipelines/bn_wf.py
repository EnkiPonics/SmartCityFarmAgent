"""Bayesian Network Workflow for aquaponics sensor data classification."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

from bayes_networks.src.jobs import (
    fit_bayesian_root_cause_model,
    score_root_causes,
    set_global_seed,
    save_bayesian_model,
    load_bayesian_model,
)


@dataclass
class BayesianNetworkClassificationResult:
    """Result of Bayesian Network classification workflow."""
    low_temp_predictions: pd.DataFrame
    low_ph_predictions: pd.DataFrame
    low_temp_probabilities: Dict[str, float]
    low_ph_probabilities: Dict[str, float]
    trace_low_temp: Optional[object] = None
    trace_low_ph: Optional[object] = None
    model_temp_path: Optional[str] = None
    model_ph_path: Optional[str] = None
    plot_temp_path: Optional[str] = None
    plot_ph_path: Optional[str] = None
    summary_path: Optional[str] = None


class BayesianNetworkClassificationWorkflow:
    """Workflow for Bayesian Network classification of aquaponics sensor data.
    
    Classifies two key conditions:
    - Low Temperature: temperature < 15°C
    - Low pH: pH_cal < 7
    """

    def __init__(
        self,
        seed: Optional[int] = 42,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        progressbar: bool = True,
    ):
        """Initialize the Bayesian Network Classification Workflow.
        
        Args:
            seed: Random seed for reproducibility
            draws: Number of posterior draws
            tune: Number of tuning steps
            chains: Number of MCMC chains
            progressbar: Show progress bar during sampling
        """
        self.seed = seed
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.progressbar = progressbar
        
        if seed is not None:
            import os
            import random
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)

    def _extract_temperature_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract temperature-related features from homogenized data."""
        features = pd.DataFrame(index=data.index)
        
        # Primary temperature sensor
        temp_cols = [col for col in data.columns if 'temp' in col.lower() or 'C' in col]
        
        if 'pressure_temperature_C' in data.columns:
            features['temp_primary'] = data['pressure_temperature_C']
        elif 'temp_sensor_93982118_C_ec' in data.columns:
            features['temp_primary'] = data['temp_sensor_93982118_C_ec']
        else:
            features['temp_primary'] = data[temp_cols[0]] if len(temp_cols) > 0 else np.nan
            
        # Backup/secondary temperature
        if 'temp_sensor_C_ph' in data.columns:
            features['temp_secondary'] = data['temp_sensor_C_ph']
        elif len(temp_cols) > 1:
            features['temp_secondary'] = data[temp_cols[1]]
        else:
            features['temp_secondary'] = features['temp_primary']
        
        # Temperature stability (moving std dev)
        features['temp_variance'] = features['temp_primary'].rolling(window=5, min_periods=1).std()
        features['temp_variance'] = features['temp_variance'].fillna(0)
        
        # Cross-sensor agreement
        features['temp_agreement'] = (np.abs(features['temp_primary'] - features['temp_secondary']) < 1.0).astype(int)
        
        # Target: Low temperature classification
        features['low_temp'] = (features['temp_primary'] < 15.0).astype(int)
        
        return features

    def _extract_ph_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract pH-related features from homogenized data."""
        features = pd.DataFrame(index=data.index)
        
        # pH calibrated value
        if 'ph_cal' in data.columns:
            features['ph_value'] = data['ph_cal']
        else:
            features['ph_value'] = np.nan
            
        # Raw pH voltage
        if 'ph_raw_mV' in data.columns:
            features['ph_raw'] = data['ph_raw_mV']
        else:
            features['ph_raw'] = np.nan
            
        # pH drift (change over time)
        features['ph_drift'] = features['ph_value'].rolling(window=5, min_periods=1).std()
        features['ph_drift'] = features['ph_drift'].fillna(0)
        
        # ORP as proxy for chemical stability
        if 'orp_raw_mV' in data.columns:
            features['orp_value'] = data['orp_raw_mV']
            features['orp_stable'] = (features['orp_value'].rolling(window=5, min_periods=1).std() < 200).astype(int)
            features['orp_stable'] = features['orp_stable'].fillna(1).astype(int)
        else:
            features['orp_value'] = np.nan
            features['orp_stable'] = 1
            
        # Target: Low pH classification
        features['low_ph'] = (features['ph_value'] < 7.0).astype(int)
        
        return features

    def _extract_system_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract system-level features that affect both temperature and pH."""
        features = pd.DataFrame(index=data.index)
        
        # EC (electrical conductivity) - affects both temp and pH sensing
        if 'ec_calibrated' in data.columns:
            features['ec_value'] = data['ec_calibrated']
            features['ec_high'] = (data['ec_calibrated'] > 1200).astype(int)
        else:
            features['ec_value'] = np.nan
            features['ec_high'] = 0
            
        # Sensor health indicators
        if 'ec_raw_uS' in data.columns and 'ec_calibrated' in data.columns:
            features['ec_sensor_health'] = (
                (data['ec_raw_uS'] > 1000) & (data['ec_raw_uS'] < 5000)
            ).astype(int)
        else:
            features['ec_sensor_health'] = 1
            
        return features

    def _prepare_classification_data(
        self,
        data: pd.DataFrame,
        target: str = 'low_temp',
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data for Bayesian Network classification.
        
        Args:
            data: Input homogenized sensor data
            target: Target classification ('low_temp' or 'low_ph')
            
        Returns:
            Processed dataframe and list of feature columns
        """
        if target == 'low_temp':
            features = self._extract_temperature_features(data)
            additional = self._extract_system_features(data)
            features = pd.concat([features, additional], axis=1)
            feature_cols = [
                'temp_variance', 'temp_agreement', 'ec_high', 'ec_sensor_health'
            ]
        elif target == 'low_ph':
            features = self._extract_ph_features(data)
            additional = self._extract_system_features(data)
            features = pd.concat([features, additional], axis=1)
            feature_cols = [
                'ph_drift', 'orp_stable', 'ec_high', 'ec_sensor_health'
            ]
        else:
            raise ValueError(f"Unknown target: {target}")
            
        # Remove rows with NaN values
        features = features.dropna(subset=[target] + feature_cols, how='any')
        
        return features, feature_cols

    def _fit_classification_model(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
    ) -> Tuple[Dict, List[str]]:
        """Fit Bayesian Network for classification using PyMC.
        
        Args:
            data: Prepared data with target and features
            target_col: Name of target column
            feature_cols: List of feature column names
            
        Returns:
            Posterior parameters and target labels
        """
        # Create cause column from target (binary: 0=not present, 1=present)
        data_model = data.copy()
        data_model['cause'] = data_model[target_col].astype(int)
        
        posterior, cause_labels = fit_bayesian_root_cause_model(
            data=data_model,
            cause_column='cause',
            symptom_columns=feature_cols,
            draws=self.draws,
            tune=self.tune,
            chains=self.chains,
            random_seed=self.seed if self.seed is not None else 42,
            progressbar=self.progressbar,
        )
        
        return posterior, cause_labels

    def _classify_observations(
        self,
        posterior: Dict,
        data: pd.DataFrame,
        feature_cols: List[str],
        cause_labels: List[str],
    ) -> pd.DataFrame:
        """Score observations for target classification.
        
        Args:
            posterior: Fitted model posteriors
            data: Input data
            feature_cols: Feature columns
            cause_labels: Labels for classification
            
        Returns:
            DataFrame with classification probabilities
        """
        predictions = []
        
        for idx, row in data.iterrows():
            symptom_values = {col: int(row[col]) for col in feature_cols}
            scores = score_root_causes(
                posterior,
                symptom_values,
                feature_cols,
                cause_labels,
            )
            scores['index'] = pd.Series([idx] * len(scores), index=scores.index, dtype=object)
            predictions.append(scores)
            
        predictions_df = pd.concat(predictions, ignore_index=True)
        return predictions_df

    def _save_model_artifacts(
        self,
        posterior: Dict,
        cause_labels: List[str],
        output_dir: str,
        model_name: str,
    ) -> str:
        """Save model artifacts to disk.
        
        Args:
            posterior: Model posteriors
            cause_labels: Classification labels
            output_dir: Output directory
            model_name: Name for the model file
            
        Returns:
            Path to saved model
        """
        output_path = Path(output_dir) / f"{model_name}_model.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_bayesian_model(posterior, cause_labels, str(output_path))
        return str(output_path)

    def _plot_predictions(
        self,
        predictions: pd.DataFrame,
        output_path: str,
        title: str,
    ) -> None:
        """Plot prediction probabilities.
        
        Args:
            predictions: Classification predictions DataFrame
            output_path: Path to save plot
            title: Plot title
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of probabilities
        ax1.hist(
            predictions[predictions['cause'] == '1']['probability'],
            bins=30,
            alpha=0.7,
            label='Target Present (cause=1)',
            color='#e74c3c',
        )
        ax1.hist(
            predictions[predictions['cause'] == '0']['probability'],
            bins=30,
            alpha=0.7,
            label='Target Absent (cause=0)',
            color='#3498db',
        )
        ax1.set_xlabel('Probability')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{title} - Probability Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Time series of max probability
        sorted_preds = predictions.sort_values('index').reset_index(drop=True)
        max_probs = sorted_preds.groupby('index')['probability'].max()
        ax2.plot(max_probs.values, linewidth=1, color='#2ecc71')
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
        ax2.fill_between(range(len(max_probs)), 0.5, max_probs.values, alpha=0.2, color='#2ecc71')
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('Maximum Probability')
        ax2.set_title(f'{title} - Confidence Over Time')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _generate_summary(
        self,
        low_temp_probs: Dict,
        low_ph_probs: Dict,
        output_path: str,
    ) -> None:
        """Generate summary report.
        
        Args:
            low_temp_probs: Low temperature probabilities
            low_ph_probs: Low pH probabilities
            output_path: Path to save summary
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'low_temperature_classification': {
                'target_threshold': '< 15°C',
                'posterior_probabilities': low_temp_probs,
            },
            'low_ph_classification': {
                'target_threshold': '< 7.0',
                'posterior_probabilities': low_ph_probs,
            },
            'workflow_parameters': {
                'seed': self.seed,
                'draws': self.draws,
                'tune': self.tune,
                'chains': self.chains,
            },
        }
        
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    def run(
        self,
        data_file: str,
        output_dir: str = './output',
        save_models: bool = True,
        save_plots: bool = True,
        save_summary: bool = True,
    ) -> BayesianNetworkClassificationResult:
        """Run the complete Bayesian Network classification workflow.
        
        Args:
            data_file: Path to homogenized sensor data CSV
            output_dir: Output directory for results
            save_models: Whether to save trained models
            save_plots: Whether to generate and save plots
            save_summary: Whether to save summary report
            
        Returns:
            BayesianNetworkClassificationResult with all outputs
        """
        print("Loading sensor data...")
        data = pd.read_csv(data_file)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Low Temperature Classification
        print("\n" + "="*60)
        print("LOW TEMPERATURE CLASSIFICATION (< 15°C)")
        print("="*60)
        
        print("Extracting temperature features...")
        temp_data, temp_features = self._prepare_classification_data(data, target='low_temp')
        print(f"  - Total samples: {len(temp_data)}")
        print(f"  - Low temperature events: {temp_data['low_temp'].sum()} ({100*temp_data['low_temp'].mean():.1f}%)")
        print(f"  - Features: {temp_features}")
        
        print("\nFitting Bayesian Network model for temperature...")
        temp_posterior, temp_labels = self._fit_classification_model(
            temp_data, 'low_temp', temp_features
        )
        
        print("Classifying observations...")
        temp_predictions = self._classify_observations(
            temp_posterior, temp_data, temp_features, temp_labels
        )
        
        # Calculate probabilities
        temp_probs = {
            str(label): float(temp_predictions[temp_predictions['cause'] == str(label)]['probability'].mean())
            for label in temp_labels
        }
        print(f"Posterior probabilities (temperature): {temp_probs}")
        
        model_temp_path = None
        if save_models:
            print("Saving temperature model...")
            model_temp_path = self._save_model_artifacts(
                temp_posterior, temp_labels, str(output_path), 'temperature'
            )
            
        plot_temp_path = None
        if save_plots:
            print("Generating temperature plots...")
            plot_temp_path = str(output_path / 'temperature_classification.png')
            self._plot_predictions(temp_predictions, plot_temp_path, 'Low Temperature')
        
        # Low pH Classification
        print("\n" + "="*60)
        print("LOW pH CLASSIFICATION (< 7.0)")
        print("="*60)
        
        print("Extracting pH features...")
        ph_data, ph_features = self._prepare_classification_data(data, target='low_ph')
        print(f"  - Total samples: {len(ph_data)}")
        print(f"  - Low pH events: {ph_data['low_ph'].sum()} ({100*ph_data['low_ph'].mean():.1f}%)")
        print(f"  - Features: {ph_features}")
        
        print("\nFitting Bayesian Network model for pH...")
        ph_posterior, ph_labels = self._fit_classification_model(
            ph_data, 'low_ph', ph_features
        )
        
        print("Classifying observations...")
        ph_predictions = self._classify_observations(
            ph_posterior, ph_data, ph_features, ph_labels
        )
        
        # Calculate probabilities
        ph_probs = {
            str(label): float(ph_predictions[ph_predictions['cause'] == str(label)]['probability'].mean())
            for label in ph_labels
        }
        print(f"Posterior probabilities (pH): {ph_probs}")
        
        model_ph_path = None
        if save_models:
            print("Saving pH model...")
            model_ph_path = self._save_model_artifacts(
                ph_posterior, ph_labels, str(output_path), 'ph'
            )
            
        plot_ph_path = None
        if save_plots:
            print("Generating pH plots...")
            plot_ph_path = str(output_path / 'ph_classification.png')
            self._plot_predictions(ph_predictions, plot_ph_path, 'Low pH')
        
        # Generate summary
        summary_path = None
        if save_summary:
            print("\nGenerating summary report...")
            summary_path = str(output_path / 'classification_summary.json')
            self._generate_summary(temp_probs, ph_probs, summary_path)
        
        print("\n" + "="*60)
        print("WORKFLOW COMPLETED")
        print("="*60)
        
        return BayesianNetworkClassificationResult(
            low_temp_predictions=temp_predictions,
            low_ph_predictions=ph_predictions,
            low_temp_probabilities=temp_probs,
            low_ph_probabilities=ph_probs,
            model_temp_path=model_temp_path,
            model_ph_path=model_ph_path,
            plot_temp_path=plot_temp_path,
            plot_ph_path=plot_ph_path,
            summary_path=summary_path,
        )


if __name__ == '__main__':
    # Example usage
    workflow = BayesianNetworkClassificationWorkflow(
        seed=42,
        draws=500,      # Reduced for faster demo
        tune=500,
        chains=2,
        progressbar=False,  # Disable on Windows to avoid encoding issues
    )
    
    data_path = Path(__file__).parent.parent / 'osi-hackathon' / 'data' / 'homogenized_sensor_data.csv'
    if not data_path.exists():
        data_path = Path(__file__).parent / 'data' / 'homogenized_sensor_data.csv'
    
    output_dir = Path(__file__).parent / 'data' / 'bn_classification_output'
    
    result = workflow.run(
        data_file=str(data_path),
        output_dir=str(output_dir),
        save_models=True,
        save_plots=True,
        save_summary=True,
    )
    
    print("\nResults:")
    print(f"Low Temperature Predictions shape: {result.low_temp_predictions.shape}")
    print(f"Low pH Predictions shape: {result.low_ph_predictions.shape}")
    print(f"\nOutput files:")
    print(f"  Model (Temp): {result.model_temp_path}")
    print(f"  Model (pH): {result.model_ph_path}")
    print(f"  Plot (Temp): {result.plot_temp_path}")
    print(f"  Plot (pH): {result.plot_ph_path}")
    print(f"  Summary: {result.summary_path}")
