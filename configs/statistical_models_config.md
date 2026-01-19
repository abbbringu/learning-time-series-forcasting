# Statistical Models Configuration

This configuration file provides recommended parameters for statistical forecasting models.

## ARIMA Models

### Non-Seasonal ARIMA
- Start with (1,1,1) and adjust based on ACF/PACF plots
- Use auto_arima from pmdarima for automatic parameter selection

```python
from pmdarima import auto_arima

model = auto_arima(
    train,
    start_p=1, start_q=1,
    max_p=5, max_q=5,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)
```

### Seasonal ARIMA (SARIMA)
- Include seasonal components: (p,d,q)(P,D,Q,s)
- Common seasonal periods: 7 (weekly), 12 (monthly), 24 (hourly), 365 (yearly)

```python
from pmdarima import auto_arima

model = auto_arima(
    train,
    seasonal=True,
    m=12,  # Seasonal period
    stepwise=True,
    suppress_warnings=True
)
```

## Prophet

### Basic Configuration
```python
from prophet import Prophet

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive'  # or 'multiplicative'
)
```

### With Additional Regressors
```python
model = Prophet()
model.add_regressor('temperature')
model.add_regressor('holiday')
```

## Exponential Smoothing

### Simple Exponential Smoothing
```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(train)
fitted = model.fit(smoothing_level=0.2, optimized=True)
```

### Holt-Winters (Triple Exponential Smoothing)
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    train,
    seasonal_periods=12,
    trend='add',
    seasonal='add'
)
fitted = model.fit()
```

## Model Selection Tips

1. **Stationary data**: Simple models (MA, AR, ARMA)
2. **Trend**: Add differencing (ARIMA with d>0) or use Holt's method
3. **Seasonality**: Use SARIMA or Holt-Winters
4. **Multiple seasonalities**: Use Prophet or TBATS
5. **Irregular patterns**: Consider ML or DL approaches
