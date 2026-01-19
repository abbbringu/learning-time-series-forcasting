"""
Evaluation metrics for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict


def mean_absolute_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    forecast : np.ndarray
        Forecasted values
    
    Returns:
    --------
    float
        MAE value
    """
    return np.mean(np.abs(actual - forecast))


def mean_squared_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE).
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    forecast : np.ndarray
        Forecasted values
    
    Returns:
    --------
    float
        MSE value
    """
    return np.mean((actual - forecast) ** 2)


def root_mean_squared_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    forecast : np.ndarray
        Forecasted values
    
    Returns:
    --------
    float
        RMSE value
    """
    return np.sqrt(mean_squared_error(actual, forecast))


def mean_absolute_percentage_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    forecast : np.ndarray
        Forecasted values
    
    Returns:
    --------
    float
        MAPE value (as percentage)
    """
    # Avoid division by zero
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100


def symmetric_mean_absolute_percentage_error(
    actual: np.ndarray,
    forecast: np.ndarray
) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    forecast : np.ndarray
        Forecasted values
    
    Returns:
    --------
    float
        sMAPE value (as percentage)
    """
    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    # Avoid division by zero
    mask = denominator != 0
    return np.mean(np.abs(actual[mask] - forecast[mask]) / denominator[mask]) * 100


def mean_absolute_scaled_error(
    actual: np.ndarray,
    forecast: np.ndarray,
    train: np.ndarray,
    seasonality: int = 1
) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    forecast : np.ndarray
        Forecasted values
    train : np.ndarray
        Training data for scaling
    seasonality : int
        Seasonal period (1 for non-seasonal)
    
    Returns:
    --------
    float
        MASE value
    """
    mae = mean_absolute_error(actual, forecast)
    
    # Calculate naive forecast error on training data
    naive_error = np.mean(np.abs(train[seasonality:] - train[:-seasonality]))
    
    return mae / naive_error if naive_error != 0 else np.inf


def r2_score(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    forecast : np.ndarray
        Forecasted values
    
    Returns:
    --------
    float
        R-squared value
    """
    ss_res = np.sum((actual - forecast) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def evaluate_forecast(
    actual: np.ndarray,
    forecast: np.ndarray,
    train: np.ndarray = None,
    seasonality: int = 1
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    forecast : np.ndarray
        Forecasted values
    train : np.ndarray, optional
        Training data (required for MASE)
    seasonality : int
        Seasonal period
    
    Returns:
    --------
    dict
        Dictionary of all metrics
    """
    metrics = {
        'MAE': mean_absolute_error(actual, forecast),
        'MSE': mean_squared_error(actual, forecast),
        'RMSE': root_mean_squared_error(actual, forecast),
        'MAPE': mean_absolute_percentage_error(actual, forecast),
        'sMAPE': symmetric_mean_absolute_percentage_error(actual, forecast),
        'R2': r2_score(actual, forecast)
    }
    
    if train is not None:
        metrics['MASE'] = mean_absolute_scaled_error(
            actual, forecast, train, seasonality
        )
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print evaluation metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    """
    print("=" * 50)
    print("Forecast Evaluation Metrics")
    print("=" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name:10s}: {value:>12.4f}")
    print("=" * 50)
