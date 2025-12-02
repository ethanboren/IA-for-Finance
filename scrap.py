import yfinance as yf
import pandas as pd
import pandas_ta as ta  # Librairie standard pour l'analyse technique

# ==========================================
# CONFIGURATION
# ==========================================
start_date = "2000-01-01"  # On étend l'historique pour avoir + de données
end_date = "2024-01-01"

print(f"1. Téléchargement des données S&P 500 et VIX ({start_date} - {end_date})...")

# Téléchargement S&P 500 (^GSPC)
sp500 = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.droplevel(1) # Nettoyage multi-index

# Téléchargement VIX (^VIX) - Indice de volatilité
vix = yf.download("^VIX", start=start_date, end=end_date, auto_adjust=True)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.droplevel(1)

# On renomme pour éviter les confusions lors de la fusion
sp500 = sp500[['Close', 'Volume', 'High', 'Low', 'Open']] # On garde tout pour les indicateurs
vix = vix[['Close']].rename(columns={'Close': 'VIX_Close'})

# Fusion des deux datasets sur la date (Index)
df = pd.merge(sp500, vix, left_index=True, right_index=True, how='inner')

print(f"   Données brutes récupérées : {df.shape[0]} lignes.")

# ==========================================
# FEATURE ENGINEERING (Augmentation des données)
# ==========================================
print("2. Calcul des indicateurs techniques (RSI, MACD, Bollinger)...")

# 1. RSI (Relative Strength Index) - Détecte surachat/survente
df['RSI'] = ta.rsi(df['Close'], length=14)

# 2. MACD (Moving Average Convergence Divergence) - Tendance
macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
df = pd.concat([df, macd], axis=1) # Ajoute les colonnes MACD

# 3. Bollinger Bands (Volatilité)
bollinger = ta.bbands(df['Close'], length=20, std=2)
df = pd.concat([df, bollinger], axis=1)

# 4. ATR (Average True Range) - Volatilité absolue
df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

# 5. Rendements Logarithmiques (La base pour le modèle)
df['Log_Ret'] = pd.NA # Init
import numpy as np
df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

# Nettoyage des NaN générés par les calculs (les 30 premières lignes seront vides)
df.dropna(inplace=True)

# ==========================================
# SAUVEGARDE
# ==========================================
filename = "sp500_full_features.csv"
df.to_csv(filename)

print(f"\nTerminé ! Fichier sauvegardé sous : {filename}")
print(f"Nouvelles dimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes")
print("-" * 30)
print("Aperçu des colonnes disponibles pour le LSTM :")
print(df.columns.tolist())