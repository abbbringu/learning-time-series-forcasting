# Machine Learning Models Configuration

Configuration and best practices for ML-based time series forecasting.

## Feature Engineering

### Lag Features
```python
from src.utils import create_lag_features

# Create lag features
lags = [1, 2, 3, 7, 14, 21, 30]  # Adjust based on your data
df_with_lags = create_lag_features(df, 'target', lags=lags)
```

### Rolling Features
```python
from src.utils import create_rolling_features

# Create rolling window statistics
windows = [7, 14, 30]
df_with_rolling = create_rolling_features(
    df, 'target',
    windows=windows,
    features=['mean', 'std', 'min', 'max']
)
```

### Time-based Features
```python
def create_time_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    return df
```

## XGBoost

### Basic Configuration
```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### With Hyperparameter Tuning
```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)
```

## LightGBM

### Basic Configuration
```python
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,
    num_leaves=31,
    min_child_samples=20,
    random_state=42
)
```

## CatBoost

### Basic Configuration
```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=False
)
```

## Random Forest

### Basic Configuration
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

## Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    # Evaluate
```

## Feature Importance

```python
import matplotlib.pyplot as plt

# Get feature importance
importance = model.feature_importances_
features = X_train.columns

# Plot
plt.figure(figsize=(10, 6))
plt.barh(features, importance)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

## Tips

1. **Start simple**: Begin with basic lag features
2. **Domain knowledge**: Add features based on your understanding
3. **Cross-validation**: Always use time series CV, not random CV
4. **Feature selection**: Remove highly correlated features
5. **Scaling**: Some models benefit from feature scaling
6. **Ensemble**: Combine multiple models for better results
