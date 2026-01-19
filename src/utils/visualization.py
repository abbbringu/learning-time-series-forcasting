"""
Visualization utilities for time series forecasting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


def plot_timeseries(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Time Series Plot",
    xlabel: str = "Date",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Plot time series data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data with datetime index
    columns : list, optional
        List of column names to plot (plots all if None)
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    if columns is None:
        columns = data.columns
    
    for col in columns:
        plt.plot(data.index, data[col], label=col, linewidth=2)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_forecast(
    actual: pd.Series,
    forecast: pd.Series,
    train: Optional[pd.Series] = None,
    confidence_intervals: Optional[pd.DataFrame] = None,
    title: str = "Forecast vs Actual",
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Plot forecast against actual values.
    
    Parameters:
    -----------
    actual : pd.Series
        Actual values
    forecast : pd.Series
        Forecasted values
    train : pd.Series, optional
        Training data to show context
    confidence_intervals : pd.DataFrame, optional
        DataFrame with 'lower' and 'upper' columns for confidence intervals
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot training data if provided
    if train is not None:
        plt.plot(train.index, train.values, label='Training Data', 
                color='blue', linewidth=2, alpha=0.7)
    
    # Plot actual and forecast
    plt.plot(actual.index, actual.values, label='Actual', 
            color='green', linewidth=2, marker='o', markersize=4)
    plt.plot(forecast.index, forecast.values, label='Forecast', 
            color='red', linewidth=2, linestyle='--', marker='s', markersize=4)
    
    # Plot confidence intervals if provided
    if confidence_intervals is not None:
        plt.fill_between(
            confidence_intervals.index,
            confidence_intervals['lower'],
            confidence_intervals['upper'],
            alpha=0.3,
            color='red',
            label='Confidence Interval'
        )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(
    residuals: pd.Series,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """
    Plot residual diagnostics.
    
    Parameters:
    -----------
    residuals : pd.Series
        Residuals (actual - predicted)
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Residuals over time
    axes[0, 0].plot(residuals.index, residuals.values, linewidth=1)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 1].hist(residuals.values, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Distribution of Residuals', fontweight='bold')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals.values, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ACF plot
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals.values, lags=min(40, len(residuals)//2), ax=axes[1, 1])
    axes[1, 1].set_title('Autocorrelation of Residuals', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_decomposition(
    data: pd.Series,
    model: str = 'additive',
    period: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Plot seasonal decomposition of time series.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    model : str
        'additive' or 'multiplicative'
    period : int, optional
        Period for seasonal decomposition
    figsize : tuple
        Figure size
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    decomposition = seasonal_decompose(data, model=model, period=period)
    
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    decomposition.observed.plot(ax=axes[0], title='Original', legend=False)
    axes[0].set_ylabel('Observed')
    
    decomposition.trend.plot(ax=axes[1], title='Trend', legend=False)
    axes[1].set_ylabel('Trend')
    
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal', legend=False)
    axes[2].set_ylabel('Seasonal')
    
    decomposition.resid.plot(ax=axes[3], title='Residual', legend=False)
    axes[3].set_ylabel('Residual')
    
    plt.tight_layout()
    plt.show()


def plot_multiple_forecasts(
    actual: pd.Series,
    forecasts: dict,
    train: Optional[pd.Series] = None,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Plot multiple forecasts for comparison.
    
    Parameters:
    -----------
    actual : pd.Series
        Actual values
    forecasts : dict
        Dictionary of {model_name: forecast_series}
    train : pd.Series, optional
        Training data
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot training data if provided
    if train is not None:
        plt.plot(train.index, train.values, label='Training Data',
                color='blue', linewidth=2, alpha=0.5)
    
    # Plot actual
    plt.plot(actual.index, actual.values, label='Actual',
            color='black', linewidth=2.5, marker='o', markersize=5)
    
    # Plot each forecast
    colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts)))
    for (model_name, forecast), color in zip(forecasts.items(), colors):
        plt.plot(forecast.index, forecast.values, label=model_name,
                linewidth=2, linestyle='--', marker='s', markersize=4, color=color)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
