import pandas as pd
import numpy as np
from pathlib import Path

from ..lstm.src.workflows import (
    LstmOnlineLearningFineTuningWorkflow,
    DataPrepConfig, TrainingConfig
)
from ..lstm.src.jobs import load_lstm_model, load_and_process_data

def lstm_online_learning_initial_training(selected):
    """Run LstmOnlineLearningFineTuningWorkflow: initial model training."""
    csv_file = "osi-hackathon/data/homogenized_sensor_data.csv"
    smoke_csv = "osi-hackathon/data/aquaponics_smoke_online_learning_initial.csv"
    plot_file = "osi-hackathon/data/aquaponics_pred_vs_expected_online_ini_predicted.png"
    parity_plot_file = "osi-hackathon/data/aquaponics_parity_plot_online_ini_predicted.png"
    scaled_plot_file = "osi-hackathon/data/aquaponics_scaled_plot_online_ini_predicted.png"
    model_file = "osi-hackathon/data/aquaponics_lstm_model_online_learning.keras"
    
    pd.read_csv(csv_file, usecols=selected).head(500).to_csv(smoke_csv, index=False)
    
    data_prep_config = DataPrepConfig(train_split=0.8, timesteps=75)
    training_config = TrainingConfig(
        optimizer="adam",
        loss="mse",
        epochs=250,
        batch_size=16,
        validation_split=0.2,
        verbose="auto",
        use_early_stopping=False
    )
    
    workflow = LstmOnlineLearningFineTuningWorkflow(
        data_prep_config=data_prep_config,
        training_config=training_config,
        fine_tune_epochs=5,
        dropout_rate=0.2,
        plot_output_path=plot_file,
        parity_plot_output_path=parity_plot_file,
        scaled_plot_output_path=scaled_plot_file,
        seed=42,
    )
    # First run: initialize and train new model
    result = workflow.run(
        file_path=str(smoke_csv),
        model_path=str(model_file),
        input_shape=(20, 1),
        target_column="ec_calibrated",
        x_axis_column="Time",
    )
    
    # Assertions
    assert result["model_existed"] is False
    assert Path(model_file).exists()
    assert result["model"] is not None
    assert isinstance(result["loss"], float)
    assert result["loss"] > 0
    assert result["model_path"] == str(model_file)
    
    # Verify predictions work
    X_train, X_test, y_train, y_test, _, _, _, _ = load_and_process_data(
        str(smoke_csv),
        data_prep_config,
        target_column="ec_calibrated",
        x_axis_column="Time",
    )
    
    predictions = result["model"].predict(X_test, verbose=0)
    assert predictions.shape[0] == X_test.shape[0]
    assert not np.isnan(predictions).any()
    
    # Cleanup
    Path(model_file).unlink()
    Path(smoke_csv).unlink()


def lstm_online_learning_fine_tuning(selected):
    """Run LstmOnlineLearningFineTuningWorkflow: load and fine-tune existing model."""
    csv_file =  "osi-hackathon/data/homogenized_sensor_data.csv"
    initial_csv = "osi-hackathon/data/aquaponics_smoke_online_learning_initial_ft.csv"
    recent_csv = "osi-hackathon/data/aquaponics_smoke_online_learning_recent_ft.csv"
    plot_file = "osi-hackathon/data/aquaponics_pred_vs_expected_online_finetune_predicted.png"
    parity_plot_file = "osi-hackathon/data/aquaponics_parity_plot_online_finetune_predicted.png"
    scaled_plot_file = "osi-hackathon/data/aquaponics_scaled_plot_online_finetune_predicted.png"
    model_file = "osi-hackathon/data/aquaponics_lstm_model_online_learning_ft.keras"
    

    full_data = pd.read_csv(csv_file, usecols=selected).head(500)
    
    # Split: first 400 for initial training, last 100 for fine-tuning
    initial_data = full_data.iloc[:400]
    recent_data = full_data.iloc[400:]
    
    initial_data.to_csv(initial_csv, index=False)
    recent_data.to_csv(recent_csv, index=False)
    
    data_prep_config = DataPrepConfig(train_split=0.8, timesteps=75)
    training_config = TrainingConfig(
        optimizer="adam",
        loss="mse",
        epochs=250,
        batch_size=16,
        validation_split=0.2,
        verbose="auto",
        use_early_stopping=False,
    )
    
    workflow = LstmOnlineLearningFineTuningWorkflow(
        data_prep_config=data_prep_config,
        training_config=training_config,
        fine_tune_epochs=5,
        dropout_rate=0.2,
        plot_output_path=plot_file,
        parity_plot_output_path=parity_plot_file,
        scaled_plot_output_path=scaled_plot_file,
        seed=42,
    )
    
    # Step 1: Initial training
    result_initial = workflow.run(
        file_path=str(initial_csv),
        model_path=str(model_file),
        input_shape=(15, 1),
        target_column="ec_calibrated",
        x_axis_column="Time",
    )
    
    assert result_initial["model_existed"] is False
    assert Path(model_file).exists()
    
    # Step 2: Fine-tune with recent data
    result_finetune = workflow.run(
        file_path=str(recent_csv),
        model_path=str(model_file),
        input_shape=(15, 1),
        target_column="ec_calibrated",
        x_axis_column="Time",
    )
    
    # Assertions
    assert result_finetune["model_existed"] is True
    assert result_finetune["model"] is not None
    assert isinstance(result_finetune["loss"], float)
    assert result_finetune["loss"] >= 0
    
    # Cleanup
    Path(model_file).unlink()
    Path(initial_csv).unlink()
    Path(recent_csv).unlink()


def lstm_online_learning_future_predictions(selected):
    """Run LstmOnlineLearningFineTuningWorkflow with future forecasting."""
    csv_file =  "osi-hackathon/data/homogenized_sensor_data.csv"
    smoke_csv = "osi-hackathon/data/aquaponics_smoke_online_learning_forecast_ft.csv"
    plot_file = "osi-hackathon/data/aquaponics_pred_vs_expected_online_future_predicted.png"
    parity_plot_file = "osi-hackathon/data/aquaponics_parity_plot_online_future_predicted.png"
    scaled_plot_file = "osi-hackathon/data/aquaponics_scaled_plot_online_future_predicted.png"
    future_plot_file = "osi-hackathon/data/aquaponics_future_plot_online_future_predicted.png"
    model_file = "osi-hackathon/data/aquaponics_lstm_model_online_learning_forecast_ft.keras"
    
    pd.read_csv(csv_file, usecols=selected).head(500).to_csv(smoke_csv, index=False)
    
    data_prep_config = DataPrepConfig(train_split=0.8, timesteps=75)
    training_config = TrainingConfig(
        optimizer="adam",
        loss="mse",
        epochs=250,
        batch_size=16,
        validation_split=0.2,
        verbose="auto",
        use_early_stopping=False,
    )
    
    workflow = LstmOnlineLearningFineTuningWorkflow(
        data_prep_config=data_prep_config,
        training_config=training_config,
        fine_tune_epochs=5,
        dropout_rate=0.2,
        plot_output_path=plot_file,
        parity_plot_output_path=parity_plot_file,
        scaled_plot_output_path=scaled_plot_file,   
        future_plot_output_path=future_plot_file,
        seed=42,
    )
    
    # Train model
    result = workflow.run(
        file_path=str(smoke_csv),
        model_path=str(model_file),
        input_shape=(15, 1),
        target_column="ec_calibrated",
        x_axis_column="Time",
    )
    
    assert result["model_existed"] is False
    
    n_future_steps = 30
    future_predictions = workflow.predict_future(
        file_path=str(smoke_csv),
        model_path=str(model_file),
        n_future_steps=n_future_steps,
        target_column="ec_calibrated",
        x_axis_column="Time",
    )
    
    # Assertions
    assert future_predictions.shape == (n_future_steps, 1)
    assert not np.isnan(future_predictions).any()
    assert not np.isinf(future_predictions).any()
    
    # Cleanup
    Path(model_file).unlink()
    Path(smoke_csv).unlink()
    Path(plot_file).unlink()
    Path(parity_plot_file).unlink()
    Path(scaled_plot_file).unlink()
    Path(future_plot_file).unlink()     


selected = ['Time', 'ec_calibrated', 'orp_raw_mV', 'ph_cal', 'pressure_temperature_C', 'temp_sensor_C_ph',"pressure_temperature_C_{SSID=\"YoungArtsDammweg_SCF\",_git=\"ffdb268\",_mac=\"48:E7:29:43:32:17\",_modbus=\"4\",_name=\"measurements\"}", 'innen_luft_C', 'Biofilter_temp_C', 'Außen_temp_C', 'dwc_temp_C']
lstm_online_learning_initial_training(selected)
lstm_online_learning_fine_tuning(selected)
lstm_online_learning_future_predictions(selected)