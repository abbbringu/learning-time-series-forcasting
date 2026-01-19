# Getting Started Example

This is a simple example to help you get started quickly with time series forecasting.

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/abbbringu/learning-time-series-forcasting.git
cd learning-time-series-forcasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use the setup script (Linux/Mac)
chmod +x setup.sh
./setup.sh
```

## 2. Create Your First Forecast

Here's a minimal example to get you started:

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Create sample data
dates = pd.date_range('2020-01-01', periods=200, freq='D')
values = np.cumsum(np.random.randn(200)) + 100
df = pd.DataFrame({'value': values}, index=dates)

# Split data
train = df[:160]
test = df[160:]

# Fit ARIMA model
model = ARIMA(train['value'], order=(1, 1, 1))
fitted = model.fit()

# Make forecast
forecast = fitted.forecast(steps=len(test))

# Evaluate
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test['value'], forecast)
print(f"MAE: {mae:.2f}")

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['value'], label='Train')
plt.plot(test.index, test['value'], label='Actual', marker='o')
plt.plot(test.index, forecast, label='Forecast', marker='s', linestyle='--')
plt.legend()
plt.title('Simple ARIMA Forecast')
plt.show()
```

## 3. Using the Utility Functions

```python
import sys
sys.path.append('src')

# Import utilities
from utils import (
    train_test_split_timeseries,
    plot_forecast,
    evaluate_forecast,
    print_metrics
)

# Split data
train, test = train_test_split_timeseries(df, test_size=0.2)

# ... train your model ...

# Evaluate with utilities
metrics = evaluate_forecast(
    actual=test['value'].values,
    forecast=forecast.values,
    train=train['value'].values
)
print_metrics(metrics)

# Visualize
plot_forecast(
    actual=test['value'],
    forecast=forecast,
    train=train['value']
)
```

## 4. Explore the Notebook Template

The easiest way to get started is with the provided notebook template:

```bash
jupyter notebook notebooks/00_forecasting_template.ipynb
```

This template includes:
- Data loading and exploration
- Statistical models (ARIMA, Prophet)
- Machine learning models (XGBoost)
- Evaluation and comparison
- Visualization

## 5. Next Steps

### For Statistical Methods:
1. Read `configs/statistical_models_config.md`
2. Try ARIMA, SARIMA, and Prophet
3. Learn about stationarity and differencing

### For Machine Learning:
1. Read `configs/ml_models_config.md`
2. Create lag and rolling features
3. Try XGBoost, LightGBM, or Random Forest

### For Deep Learning:
1. Read `configs/dl_models_config.md`
2. Implement LSTM or GRU models
3. Explore transformer-based approaches

## 6. Common Datasets to Practice

### Built-in Datasets (via libraries)
```python
# Airline passengers (classic dataset)
from statsmodels.datasets import co2
data = co2.load().data

# Energy consumption
from sktime.datasets import load_airline
data = load_airline()
```

### External Datasets
- [Kaggle Time Series Datasets](https://www.kaggle.com/datasets?tags=13206-time+series)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [M4 Competition Data](https://www.m4.unic.ac.cy/)

## 7. Troubleshooting

### Dependencies Not Installing?
Try installing in smaller groups:
```bash
pip install numpy pandas matplotlib seaborn
pip install statsmodels prophet pmdarima
pip install scikit-learn xgboost lightgbm
```

### Import Errors in Notebooks?
Make sure you're running Jupyter from the correct environment:
```bash
source venv/bin/activate  # Activate environment first
jupyter notebook
```

### Utilities Not Found?
Check that you're adding the src directory to path:
```python
import sys
sys.path.append('../src')  # In notebooks
# or
sys.path.append('src')  # In scripts
```

## 8. Learning Resources

- **Book**: [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- **Course**: [Time Series Analysis on Coursera](https://www.coursera.org/learn/practical-time-series-analysis)
- **Documentation**: See QUICK_REFERENCE.md for code snippets

## Need Help?

- Check the configuration files in `configs/`
- Review the example notebook in `notebooks/`
- Consult the QUICK_REFERENCE.md for common patterns

Happy forecasting! 🚀📈
