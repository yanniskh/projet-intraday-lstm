# FULL CODE SANS STREAMLIT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

### Partie CHARGEMENT DES DONNÉES
df = pd.read_csv(r"C:\Users\yanni\Desktop\Python\DERIVATIVES\CNAM\Data\Processed\LVMH tick price 2024.csv", sep=';')
df['Trading Date Time'] = pd.to_datetime(df['Trading Date Time'])

df = df.set_index('Trading Date Time')
df_5min = df.resample('5min').agg({ 
    'Price': ['first', 'max', 'min', 'last'],
    'Volume': 'sum'
})
df_5min = df_5min.dropna()

    # Aplatir les colonnes 
df_5min.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
df_5min = df_5min.dropna()

print(f"Ticks chargés : {len(df):,}")
print(f"Période : {df.index.min()} → {df.index.max()}")
print(df_5min.head())
print(f"Bougies 5min : {len(df_5min)}")

### Partie CALCUL DES FEATURES
    # Rendement sur la bougie précédente
df_5min['Return_5m (%)']  = np.log(df_5min['Close'] / df_5min['Close'].shift(1)) * 100 #enlever le *100 pour avoir en décimal plutôt qu'en pourcentage

    # Rendement sur les 15 dernières minutes (3 bougies)
 df_5min['Return_15m (%)'] = np.log(df_5min['Close'] / df_5min['Close'].shift(3)) * 100 #enlever le *100 pour avoir en décimal plutôt qu'en pourcentage

    # High-Low spread (volatilité intra-bougie)
df_5min['HL_Spread (%)'] = ((df_5min['High'] - df_5min['Low']) / df_5min['Close']) * 100 #enlever le *100 pour avoir en décimal plutôt qu'en pourcentage

    # Ratio volume vs moyenne mobile 20 périodes
df_5min['Volume_Ratio'] = df_5min['Volume'] / df_5min['Volume'].rolling(20).mean()

    # Heure de la journée (feature temporelle - ça nous permettra de savoir ou non quand le marché est plus actif - utile pour le LSTM)
df_5min['Hour'] = df_5min.index.hour + df_5min.index.minute / 60

df_5min = df_5min.dropna()
print(f"Bougies après feature engineering : {len(df_5min)}")
print(df_5min.head())

# Variable cible pour LSTM (plus tard) : prédire si le prix va monter ou descendre dans les 5 prochaines minutes
    # 1 si le Close dans 5 minutes est plus haut qu'aujourd'hui, 0 sinon
df_5min['Target'] = (df_5min['Close'].shift(-1) > df_5min['Close']).astype(int)

    # Supprimer la dernière ligne (pas de target possible)
df_5min = df_5min.dropna()

print(f"Distribution target :\n{df_5min['Target'].value_counts()}")
