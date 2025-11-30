import yaml
import logging
import os
import pandas as pd
import json

logger = logging.getLogger(__name__)

def load_config(config_path):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML config file: {exc}")
        raise

def save_config(config, config_path):
    """Saves configuration to a YAML file."""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        logger.error(f"Error saving config file: {e}")
        raise

def log_experiment_results(log_filepath, experiment_data):
    """
    Logs experiment configuration and results to a CSV file.
    
    Args:
        log_filepath (str): Path to the CSV file where results will be logged.
        experiment_data (dict): A dictionary containing experiment details,
                                config parameters, and final metrics.
    """
    # Convert experiment_data to a DataFrame
    df_new_entry = pd.DataFrame([experiment_data])

    # Check if the log file exists
    if not os.path.exists(log_filepath):
        # If not, create it with header
        df_new_entry.to_csv(log_filepath, mode='w', header=True, index=False)
        logger.info(f"Created new experiment log file: {log_filepath}")
    else:
        # If it exists, append without header
        df_new_entry.to_csv(log_filepath, mode='a', header=False, index=False)
        logger.info(f"Appended results to experiment log file: {log_filepath}")