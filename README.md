# Learning Time Series Forecasting

A comprehensive learning environment for exploring different time series forecasting approaches, from statistical methods to machine learning and transformer-based models.

## 📚 What You'll Learn

- **Statistical Methods**: ARIMA, SARIMA, Exponential Smoothing, Prophet
- **Machine Learning**: Random Forest, XGBoost, LightGBM, CatBoost
- **Deep Learning**: LSTM, GRU, CNN-based models
- **Transformers**: Attention-based models for time series

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/abbbringu/learning-time-series-forcasting.git
cd learning-time-series-forcasting
```

### 2. Set Up Python Environment

#### Option A: Using venv (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using conda
```bash
# Create conda environment
conda create -n forecasting python=3.10
conda activate forecasting

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Jupyter
```bash
jupyter notebook
```

## 📁 Project Structure

```
learning-time-series-forcasting/
├── data/
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned and transformed data
│   └── external/         # External datasets and references
├── notebooks/            # Jupyter notebooks for experiments
├── projects/             # Individual forecasting projects
├── src/
│   ├── utils/           # Utility functions (data loading, visualization)
│   └── models/          # Custom model implementations
├── configs/             # Configuration files for different approaches
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 📊 Available Libraries

### Statistical Forecasting
- **statsmodels**: ARIMA, SARIMA, Exponential Smoothing
- **pmdarima**: Auto-ARIMA model selection
- **prophet**: Facebook's forecasting tool

### Machine Learning
- **scikit-learn**: Traditional ML algorithms
- **xgboost**, **lightgbm**, **catboost**: Gradient boosting libraries

### Deep Learning & Transformers
- **PyTorch** & **TensorFlow**: Deep learning frameworks
- **transformers**: Hugging Face transformers library

### Time Series Specific
- **sktime**: Unified time series ML framework
- **tslearn**: Time series clustering and classification
- **darts**: Forecasting with statistical and deep learning models
- **neuralforecast**: Neural forecasting models

## 🎯 Learning Path

### Beginner
1. Start with statistical methods (ARIMA, Prophet)
2. Understand data preprocessing and feature engineering
3. Learn evaluation metrics (MAE, RMSE, MAPE)

### Intermediate
4. Explore machine learning approaches (XGBoost, LightGBM)
5. Implement cross-validation for time series
6. Feature engineering with lag features and rolling statistics

### Advanced
7. Deep learning models (LSTM, GRU)
8. Transformer-based architectures
9. Ensemble methods and model stacking

## 💡 Getting Started with Examples

Check out the `notebooks/` directory for example notebooks covering:
- Data exploration and visualization
- Statistical forecasting methods
- ML-based forecasting
- Deep learning approaches
- Transformer models

## 🔧 Utility Functions

The `src/utils/` directory contains helper functions for:
- Data loading and preprocessing
- Visualization (time series plots, forecast comparisons)
- Evaluation metrics
- Feature engineering

## 📝 Working on Projects

Create a new project folder under `projects/` for each forecasting problem:
```bash
projects/
├── sales-forecasting/
├── energy-demand/
└── stock-prediction/
```

## 🤝 Contributing

This is a personal learning repository, but suggestions and improvements are welcome!

## 📖 Resources

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [Time Series Analysis with Python](https://www.machinelearningplus.com/time-series/time-series-analysis-python/)
- [Awesome Time Series](https://github.com/cuge1995/awesome-time-series)

## 📄 License

MIT License - Feel free to use this for your own learning!