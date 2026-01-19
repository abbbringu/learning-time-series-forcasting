# Data Directory

This directory contains all datasets for your forecasting projects.

## Structure

- **raw/**: Original, immutable data dumps
- **processed/**: Cleaned and transformed datasets ready for modeling
- **external/**: External datasets and reference data

## Usage

### Raw Data
Place your original datasets in the `raw/` directory. These should never be modified directly.

### Processed Data
Store cleaned and preprocessed data in the `processed/` directory. This data is ready for modeling.

### External Data
Store external reference datasets, lookup tables, or third-party data in the `external/` directory.

## Example Datasets

Here are some public datasets you can use for learning:

1. **AirPassengers**: Classic airline passenger data
2. **Electricity**: Hourly electricity consumption
3. **Rossmann Store Sales**: Daily sales data
4. **M4 Competition**: Multiple time series datasets
5. **Energy Demand**: Power consumption data

## Data Sources

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Time Series Data Library](https://pkg.yangzhuoranyang.com/tsdl/)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

## Note

Large data files (CSV, Parquet, etc.) are excluded from git via `.gitignore`. Keep this in mind when sharing your projects.
