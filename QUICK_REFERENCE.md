# Quick Reference Guide

A quick reference for common time series forecasting tasks.

## Getting Started

### 1. Load Data
```python
from src.utils import load_csv_timeseries

df = load_csv_timeseries(
    filepath='data/raw/your_data.csv',
    date_column='date',
    target_column='value',
    freq='D'
)
```

### 2. Train/Test Split
```python
from src.utils import train_test_split_timeseries

train, test = train_test_split_timeseries(df, test_size=0.2)
```

### 3. Visualize
```python
from src.utils import plot_timeseries

plot_timeseries(df, title='My Time Series')
```

## Statistical Models

### ARIMA
```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train['value'], order=(1, 1, 1))
fitted = model.fit()
forecast = fitted.forecast(steps=len(test))
```

### Auto ARIMA
```python
from pmdarima import auto_arima

model = auto_arima(train['value'], seasonal=False, stepwise=True)
forecast = model.predict(n_periods=len(test))
```

### Prophet
```python
from prophet import Prophet

train_prophet = train.reset_index()
train_prophet.columns = ['ds', 'y']

model = Prophet()
model.fit(train_prophet)

future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)
```

## Machine Learning

### Feature Engineering
```python
from src.utils import create_lag_features, create_rolling_features

# Create lag features
df = create_lag_features(df, 'value', lags=[1, 7, 14, 30])

# Create rolling features
df = create_rolling_features(df, 'value', windows=[7, 30])
```

### XGBoost
```python
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Deep Learning

### LSTM (Keras)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

## Evaluation

### Calculate Metrics
```python
from src.utils import evaluate_forecast, print_metrics

metrics = evaluate_forecast(
    actual=test['value'].values,
    forecast=predictions,
    train=train['value'].values
)
print_metrics(metrics)
```

### Plot Forecast
```python
from src.utils import plot_forecast

plot_forecast(
    actual=test['value'],
    forecast=predictions,
    train=train['value']
)
```

### Compare Models
```python
from src.utils import plot_multiple_forecasts

forecasts = {
    'ARIMA': arima_forecast,
    'Prophet': prophet_forecast,
    'XGBoost': xgb_forecast
}

plot_multiple_forecasts(
    actual=test['value'],
    forecasts=forecasts,
    train=train['value']
)
```

## Common Patterns

### Check Stationarity
```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['value'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
# If p-value > 0.05, series is non-stationary
```

### Make Series Stationary
```python
# Differencing
df['value_diff'] = df['value'].diff()

# Log transformation
df['value_log'] = np.log(df['value'])
```

### Seasonal Decomposition
```python
from src.utils import plot_decomposition

plot_decomposition(df['value'], model='additive', period=365)
```

## Tips

1. **Start simple**: Begin with ARIMA or Prophet
2. **Visualize first**: Always plot your data before modeling
3. **Check stationarity**: Use ADF test for ARIMA models
4. **Feature engineering**: Critical for ML models
5. **Cross-validate**: Use time series CV, not random split
6. **Ensemble**: Combine multiple models for better results
7. **Domain knowledge**: Use your understanding of the data
8. **Iterate**: Try multiple approaches and compare

## Resources

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [statsmodels Documentation](https://www.statsmodels.org/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [sktime Documentation](https://www.sktime.net/)
