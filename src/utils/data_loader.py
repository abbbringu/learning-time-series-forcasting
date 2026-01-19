"""
Utility functions for loading and preprocessing time series data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_csv_timeseries(
    filepath: str,
    date_column: str = 'date',
    target_column: str = 'value',
    parse_dates: bool = True,
    freq: Optional[str] = None
) -> pd.DataFrame:
    """
    Load time series data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    date_column : str
        Name of the date/datetime column
    target_column : str
        Name of the target variable column
    parse_dates : bool
        Whether to parse dates
    freq : str, optional
        Frequency to set for the time series (e.g., 'D', 'H', 'M')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index
    """
    df = pd.read_csv(filepath, parse_dates=[date_column] if parse_dates else None)
    df = df.set_index(date_column)
    
    if freq:
        df = df.asfreq(freq)
    
    return df


def train_test_split_timeseries(
    data: pd.DataFrame,
    test_size: float = 0.2,
    gap: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    test_size : float
        Proportion of data to use for testing (0 to 1)
    gap : int
        Number of periods to skip between train and test
    
    Returns:
    --------
    tuple
        (train_data, test_data)
    """
    split_idx = int(len(data) * (1 - test_size))
    train = data.iloc[:split_idx - gap]
    test = data.iloc[split_idx:]
    
    return train, test


def create_lag_features(
    data: pd.DataFrame,
    target_col: str,
    lags: list,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Create lag features for time series forecasting.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    target_col : str
        Name of the target column
    lags : list
        List of lag periods to create
    dropna : bool
        Whether to drop rows with NaN values
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with lag features
    """
    df = data.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    if dropna:
        df = df.dropna()
    
    return df


def create_rolling_features(
    data: pd.DataFrame,
    target_col: str,
    windows: list,
    features: list = ['mean', 'std', 'min', 'max']
) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    target_col : str
        Name of the target column
    windows : list
        List of window sizes
    features : list
        List of aggregation functions to apply
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling features
    """
    df = data.copy()
    
    for window in windows:
        for feature in features:
            col_name = f'{target_col}_rolling_{window}_{feature}'
            df[col_name] = df[target_col].rolling(window=window).agg(feature)
    
    return df


def detect_outliers_iqr(
    data: pd.Series,
    multiplier: float = 1.5
) -> pd.Series:
    """
    Detect outliers using the IQR method.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    multiplier : float
        IQR multiplier for determining outliers
    
    Returns:
    --------
    pd.Series
        Boolean series indicating outliers (True = outlier)
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (data < lower_bound) | (data > upper_bound)


def fill_missing_values(
    data: pd.DataFrame,
    method: str = 'linear',
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Fill missing values in time series data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data with missing values
    method : str
        Interpolation method ('linear', 'ffill', 'bfill', 'mean')
    limit : int, optional
        Maximum number of consecutive NaNs to fill
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with filled values
    """
    df = data.copy()
    
    if method == 'mean':
        df = df.fillna(df.mean())
    elif method in ['linear', 'time', 'polynomial']:
        df = df.interpolate(method=method, limit=limit)
    elif method in ['ffill', 'bfill']:
        df = df.fillna(method=method, limit=limit)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df
