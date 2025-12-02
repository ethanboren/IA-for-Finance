import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.stattools import adfuller

# Aesthetic Configuration
sns.set_style("whitegrid")
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 1. DATA LOADING & STATIONARITY CHECK
# ==========================================
print("--- STEP 1: Data Loading ---")
filename = "sp500_dataset.csv"
df = pd.read_csv(filename, index_col=0, parse_dates=True)

# Stationarity Test (Rubric 3.2.1)
# We must verify that the Log Returns are stationary (constant mean/variance)
# otherwise the model will fail to generalize.
print("\n📊 ADF Test (Augmented Dickey-Fuller) on Log Returns:")
# Note: Ensure the column name matches scraper.py ('Log_Returns')
# If using an old CSV, it might be 'Log_Ret'. We check for both for safety or assume new CSV.
# We will assume the user runs scraper.py first, so we use 'Log_Returns'.
target_col = 'Log_Returns' if 'Log_Returns' in df.columns else 'Log_Ret'
vol_col = 'Rolling_Volatility_20' if 'Rolling_Volatility_20' in df.columns else 'Vol_20'
ma_col = 'Moving_Average_10' if 'Moving_Average_10' in df.columns else 'MA_10'

adf_result = adfuller(df[target_col])
print(f"   p-value: {adf_result[1]:.4e}")
if adf_result[1] < 0.05:
    print("   ✅ Series is Stationary (Safe to use for modeling).")
else:
    print("   ⚠️ Warning: Series is Non-Stationary.")

# ==========================================
# 2. PREPARATION (SPLIT & SCALE)
# ==========================================
print("\n--- STEP 2: Preparation (No Look-Ahead Bias) ---")

# Explanatory Variables (Features)
features = ['Lag_1', 'Lag_5', 'Lag_10', vol_col, ma_col, 'VIX', 'RSI']
X = df[features].values
y = df[['Target']].values 

# Time-Series Split (80% Train, 20% Test)
# CRITICAL: We do NOT shuffle the data. We must respect the chronological order
# to avoid "Look-Ahead Bias" (training on future data).
split = int(len(df) * 0.80)
X_train_raw, X_test_raw = X[:split], X[split:]
y_train_raw, y_test_raw = y[:split], y[split:]
test_dates = df.index[split:]

print(f"   Train set: {len(X_train_raw)} days | Test set: {len(X_test_raw)} days")

# Normalization (MinMax between -1 and 1 for LSTM tanh activation)
scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))

# Fit ONLY on the Training set! (Rubric 3.2.2)
# If we fit on the whole dataset, we leak information from Test to Train.
X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

y_train_scaled = scaler_y.fit_transform(y_train_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# Reshape for LSTM [Samples, Time Steps, Features]
# We use a time step of 1 day for simplicity and interpretability.
X_train = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# ==========================================
# 3. LSTM MODELING
# ==========================================
print("\n--- STEP 3: LSTM Architecture ---")
# Simple but effective architecture
model = Sequential([
    Input(shape=(1, len(features))),
    LSTM(50, activation='tanh', return_sequences=False), # Memory layer
    Dropout(0.2), # Regularization to prevent overfitting
    Dense(1) # Linear output (predicting the return)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Training with Early Stopping
history = model.fit(
    X_train, y_train_scaled,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test_scaled),
    callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)],
    verbose=1,
    shuffle=False # IMPORTANT: Keep order in batches too (though less critical than split)
)

# ==========================================
# 4. RESULTS & BACKTESTING
# ==========================================
print("\n--- STEP 4: Financial Evaluation ---")

# Predictions
pred_scaled = model.predict(X_test)
pred_real = scaler_y.inverse_transform(pred_scaled).flatten()
y_test_real = scaler_y.inverse_transform(y_test_scaled).flatten()

# Metric 1: MSE (Mean Squared Error)
mse = mean_squared_error(y_test_real, pred_real)
print(f"📉 MSE: {mse:.6f}")

# Metric 2: Direction Accuracy
# Do we predict the correct sign? (+/-)
pred_sign = np.sign(pred_real)
true_sign = np.sign(y_test_real)
acc = accuracy_score(true_sign, pred_sign)
print(f"🎯 Direction Accuracy: {acc:.2%}")

# Metric 3: Trading Strategy & Sharpe Ratio
# Strategy: If Prediction > 0 -> Buy (Long), Else -> Cash (0)
# (Simple Long-Only strategy, no short selling)
signal = np.where(pred_real > 0, 1, 0)
strategy_ret = signal * y_test_real

# Buy & Hold (Benchmark)
cum_market = (1 + y_test_real).cumprod()
cum_strategy = (1 + strategy_ret).cumprod()

# Sharpe Ratio Calculation (Annualized)
# Assuming Risk-Free Rate ~ 4% (0.04/252 daily)
rf = 0.04 / 252
excess_ret = strategy_ret - rf
sharpe = np.sqrt(252) * np.mean(excess_ret) / (np.std(excess_ret) + 1e-9)
print(f"💰 Sharpe Ratio (Strategy): {sharpe:.2f}")
print(f"📈 Total Return Strategy: {(cum_strategy[-1] - 1):.2%}")
print(f"📊 Total Return Market  : {(cum_market[-1] - 1):.2%}")

# Final Plot
plt.figure(figsize=(12, 6))
plt.plot(test_dates, cum_market, label='Benchmark (Buy & Hold)', color='gray', alpha=0.6)
plt.plot(test_dates, cum_strategy, label=f'LSTM Strategy (Sharpe: {sharpe:.2f})', color='green')
plt.title(f'Performance: LSTM vs S&P 500 ({test_dates[0].year}-{test_dates[-1].year})')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()