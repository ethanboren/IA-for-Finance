import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import datetime

# ==========================================
# 1. CONFIGURATION & DOWNLOAD
# ==========================================
# We fetch a long history (from 2000) to ensure sufficient data for training
start_date = "2000-01-01"
end_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
filename = "sp500_dataset.csv"

print(f"📥 Downloading S&P 500 and VIX data ({start_date} to {end_date})...")

# S&P 500 (^GSPC)
sp500 = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.droplevel(1)

# VIX (^VIX) - Exogenous Variable (Market Fear Index)
vix = yf.download("^VIX", start=start_date, end=end_date, auto_adjust=True)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.droplevel(1)

# Merge on Date index
df = pd.merge(sp500[['Close', 'Volume']], vix[['Close']], 
              left_index=True, right_index=True, suffixes=('', '_VIX'), how='inner')
df.rename(columns={'Close_VIX': 'VIX'}, inplace=True)

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("⚙️ Calculating financial indicators...")

# A. Log Returns (Essential for Stationarity)
# Formula: R_t = ln(P_t / P_{t-1})
# We use Log Returns instead of prices because prices are non-stationary (trends).
df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

# B. Lagged Returns (1, 5, 10 days)
# Past returns used as features to predict future returns.
df['Lag_1'] = df['Log_Returns'].shift(1)
df['Lag_5'] = df['Log_Returns'].shift(5)
df['Lag_10'] = df['Log_Returns'].shift(10)

# C. Rolling Volatility (20-day Risk)
# Standard deviation of returns over the past month (~20 trading days).
df['Rolling_Volatility_20'] = df['Log_Returns'].rolling(window=20).std()

# D. Moving Averages (Trend)
# Simple Moving Average of returns to capture short-term momentum.
df['Moving_Average_10'] = df['Log_Returns'].rolling(window=10).mean()

# E. RSI (Relative Strength Index)
# Technical momentum indicator.
df['RSI'] = ta.rsi(df['Close'], length=14)

# F. TARGET VARIABLE: Next Day's Return
# We shift the Log Returns backwards by 1 to align today's features with tomorrow's return.
df['Target'] = df['Log_Returns'].shift(-1)

# ==========================================
# 3. CLEANUP & SAVE
# ==========================================
# Remove NaN values created by lags and rolling windows
df_clean = df.dropna()

# Keep only relevant columns for the model
cols_to_keep = ['Close', 'Log_Returns', 'Target', 'Lag_1', 'Lag_5', 'Lag_10', 
                'Rolling_Volatility_20', 'Moving_Average_10', 'VIX', 'RSI']
df_final = df_clean[cols_to_keep]

df_final.to_csv(filename)

print(f"\n✅ Done! Dataset saved to: {filename}")
print(f"📊 Final Dimensions: {df_final.shape[0]} trading days.")
print(f"📌 Columns: {df_final.columns.tolist()}")