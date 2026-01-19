# Deep Learning & Transformer Models Configuration

Configuration and best practices for deep learning and transformer-based forecasting.

## LSTM (Long Short-Term Memory)

### Basic PyTorch Implementation
```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state
        out = self.fc(out[:, -1, :])
        return out

# Training configuration
model = LSTMForecaster(input_size=1, hidden_size=50, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### TensorFlow/Keras Implementation
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

## GRU (Gated Recurrent Unit)

### PyTorch Implementation
```python
class GRUForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(GRUForecaster, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
```

## CNN for Time Series

### 1D CNN Implementation
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
```

## Transformer Models

### Using Darts Library
```python
from darts.models import TransformerModel

model = TransformerModel(
    input_chunk_length=24,
    output_chunk_length=12,
    d_model=64,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=256,
    dropout=0.1,
    activation='relu',
    random_state=42
)

model.fit(train_series, epochs=100, verbose=True)
forecast = model.predict(n=12)
```

### Using NeuralForecast
```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS

models = [
    NBEATS(input_size=24, h=12, max_steps=1000),
    NHITS(input_size=24, h=12, max_steps=1000)
]

nf = NeuralForecast(models=models, freq='D')
nf.fit(df=train_df)
forecasts = nf.predict()
```

## Data Preparation for Deep Learning

### Sequence Creation
```python
import numpy as np

def create_sequences(data, seq_length):
    """Create sequences for LSTM/GRU training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Usage
seq_length = 30  # Use last 30 days to predict next day
X, y = create_sequences(train_data.values, seq_length)
```

### Data Normalization
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.reshape(-1, 1))

# After prediction, inverse transform
predictions = scaler.inverse_transform(predictions)
```

## Training Tips

### Early Stopping
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stop],
    verbose=1
)
```

### Learning Rate Scheduling
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)
```

## Model Selection Guide

| Model | Best For | Complexity | Training Time |
|-------|----------|------------|---------------|
| LSTM | Long-term dependencies | Medium | Medium |
| GRU | Faster alternative to LSTM | Medium | Fast |
| CNN | Pattern recognition | Low | Fast |
| Transformer | Complex patterns, long sequences | High | Slow |
| N-BEATS | General forecasting | Medium | Medium |
| Temporal Fusion Transformer | Multi-horizon, interpretable | High | Slow |

## Best Practices

1. **Sequence length**: Start with 1-2 seasonal periods
2. **Normalization**: Always normalize/standardize your data
3. **Validation**: Use a separate validation set for early stopping
4. **Overfitting**: Use dropout (0.2-0.5) and regularization
5. **Batch size**: Start with 32-64, adjust based on data size
6. **Learning rate**: Start with 0.001, use schedulers
7. **Architecture**: Start simple, add complexity if needed
8. **Ensembles**: Average predictions from multiple models
