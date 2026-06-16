import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import anthropic

# Page de configuration
st.set_page_config(page_title="Intraday Price Prediction", layout="wide")
st.title("📈 Intraday Price Prediction")
st.caption("Logistic Regression vs Random Forest vs LSTM")

# Barre latérale pour les paramètres
st.sidebar.header("Paramètres")
ticker  = st.sidebar.text_input("Ticker", value="AAPL")
period  = st.sidebar.selectbox("Période", ["30d", "60d"], index=1)
epochs  = st.sidebar.slider("Epochs LSTM", min_value=10, max_value=60, value=40, step=5)
if st.sidebar.button("🚀 Lancer l'analyse"):
    st.session_state.analysed = True

if not st.session_state.get('analysed', False):
    st.info("Configurer les paramètres dans la sidebar et clique sur Lancer l'analyse.")
    st.stop()

# Chargement
with st.spinner("Chargement des données..."):
    df = yf.download(ticker, period=period, interval="5m", progress=False)
    df = df.dropna()
    df.columns = df.columns.get_level_values(0)
    df.columns.name = None
    df = df[['Close', 'High', 'Low', 'Open', 'Volume']]

st.success(f"{len(df):,} bougies chargées — {df.index.min().date()} → {df.index.max().date()}") 

# Onglets pour les différentes sections
tab1, tab2, tab3, tab4 = st.tabs(["📊 Données", "🤖 Modèles", "💰 Backtest", "📋 Résumé"])

# Onglet 1 - DONNÉES
with tab1:
    st.subheader("Prix de clôture")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Close'], color='#4C72B0', linewidth=0.8)
    ax.set_title(f"{ticker} — Close Price")
    ax.set_ylabel("Prix ($)")
    st.pyplot(fig)
    plt.close()

    st.subheader("Feature Engineering")

    # Features
    df['Return_5m (%)']  = np.log(df['Close'] / df['Close'].shift(1)) * 100
    df['Return_15m (%)'] = np.log(df['Close'] / df['Close'].shift(3)) * 100
    df['HL_Spread (%)']  = ((df['High'] - df['Low']) / df['Close']) * 100
    df['Volume_Ratio']   = df['Volume'] / df['Volume'].rolling(20).mean()
    df['Hour']           = df.index.hour + df.index.minute / 60
    df = df.dropna()

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    col1, col2, col3 = st.columns(3)
    col1.metric("Bougies totales", f"{len(df):,}")
    col2.metric("Baisses (0)", df['Target'].value_counts()[0])
    col3.metric("Hausses (1)", df['Target'].value_counts()[1])

    st.dataframe(df[['Close', 'Return_5m (%)', 'Return_15m (%)', 'HL_Spread (%)', 'Volume_Ratio', 'Hour', 'Target']].tail(10))

# PRÉPARATION ML (commune aux onglets 2 et 3)
features = ['Return_5m (%)', 'Return_15m (%)', 'HL_Spread (%)', 'Volume_Ratio', 'Hour']
X = df[features]
y = df['Target']

split_train = int(len(df) * 0.70)
split_val   = int(len(df) * 0.85)

X_train = X.iloc[:split_train]
X_val   = X.iloc[split_train:split_val]
X_test  = X.iloc[split_val:]
y_train = y.iloc[:split_train]
y_val   = y.iloc[split_train:split_val]
y_test  = y.iloc[split_val:]

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_sc, y_train)
y_pred_lr = lr_model.predict(X_test_sc)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_sc, y_train)
y_pred_rf = rf_model.predict(X_test_sc)

# LSTM
SEQUENCE_LENGTH = 20

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_all_sc = scaler.transform(df[features])
y_all    = df['Target'].values
X_seq, y_seq = create_sequences(X_all_sc, y_all, SEQUENCE_LENGTH)

n         = len(X_seq)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train_seq, y_train_seq = X_seq[:train_end], y_seq[:train_end]
X_val_seq,   y_val_seq   = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test_seq,  y_test_seq  = X_seq[val_end:], y_seq[val_end:]

X_train_t = torch.FloatTensor(X_train_seq)
y_train_t = torch.LongTensor(y_train_seq)
X_val_t   = torch.FloatTensor(X_val_seq)
y_val_t   = torch.LongTensor(y_val_seq)
X_test_t  = torch.FloatTensor(X_test_seq)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, 2)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

torch.manual_seed(13)
np.random.seed(13)
model_lstm = LSTMModel(input_size=len(features))
criterion  = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model_lstm.parameters(), lr=0.001)

with st.spinner("Entraînement du LSTM..."):
    for epoch in range(epochs):
        model_lstm.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model_lstm(X_batch), y_batch)
            loss.backward()
            optimizer.step()

model_lstm.eval()
with torch.no_grad():
    test_preds = torch.argmax(model_lstm(X_test_t), dim=1).numpy()

acc_lr   = accuracy_score(y_test, y_pred_lr)
acc_rf   = accuracy_score(y_test, y_pred_rf)
acc_lstm = accuracy_score(y_test_seq, test_preds)

# Onglet 2 - MODÈLES
with tab2:
    st.subheader("Comparaison des modèles")

    col1, col2, col3 = st.columns(3)
    col1.metric("Logistic Regression", f"{acc_lr:.2%}")
    col2.metric("Random Forest",       f"{acc_rf:.2%}")
    col3.metric("LSTM",                f"{acc_lstm:.2%}", delta=f"+{acc_lstm - acc_lr:.2%} vs LR")

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(['Logistic Regression', 'Random Forest', 'LSTM'],
                  [acc_lr, acc_rf, acc_lstm],
                  color=['#4C72B0', '#55A868', '#C44E52'])
    ax.axhline(y=0.5, color='black', linestyle='--', label='Baseline 50%')
    ax.set_ylim(0.48, 0.60)
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy — Comparaison des modèles")
    ax.legend()
    for bar, acc in zip(bars, [acc_lr, acc_rf, acc_lstm]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{acc:.2%}', ha='center', fontweight='bold')
    st.pyplot(fig)
    plt.close()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Matrice de confusion — LSTM")
        cm  = confusion_matrix(y_test_seq, test_preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=['Baisse', 'Hausse']).plot(ax=ax, colorbar=False)
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Feature Importance — Random Forest")
        feat_imp = pd.Series(rf_model.feature_importances_, index=features).sort_values()
        fig, ax  = plt.subplots(figsize=(5, 4))
        feat_imp.plot(kind='barh', color='#55A868', ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)
        plt.close()
    
    st.subheader("Analyse de Sensibilité — Impact du retrait de chaque feature")
    
    sensitivity_results = []
    for feat in features:
        features_reduced = [f for f in features if f != feat]
        scaler_tmp  = StandardScaler()
        X_train_tmp = scaler_tmp.fit_transform(X_train[features_reduced])
        X_test_tmp  = scaler_tmp.transform(X_test[features_reduced])
        lr_tmp      = LogisticRegression(max_iter=1000)
        lr_tmp.fit(X_train_tmp, y_train)
        acc_tmp = accuracy_score(y_test, lr_tmp.predict(X_test_tmp))
        sensitivity_results.append({
            'Feature retirée': feat,
            'Accuracy': f"{acc_tmp:.2%}",
            'Delta vs baseline': f"{acc_tmp - acc_lr:+.2%}"
        })
    
    st.dataframe(pd.DataFrame(sensitivity_results), hide_index=True)

# Onglet 3 - BACKTEST

with tab3:
    st.subheader("Backtest — Stratégie LSTM vs Buy & Hold")

    close_test = df['Close'].values[len(df) - len(y_test_seq):]
    backtest   = pd.DataFrame({
        'Close' : close_test,
        'Signal': test_preds
    }, index=df.index[len(df) - len(y_test_seq):])

    TRANSACTION_COST = 0.0001
    backtest['Return_Market']   = np.log(backtest['Close'] / backtest['Close'].shift(1))
    backtest['Return_Strategy'] = backtest['Return_Market'] * backtest['Signal'].shift(1)
    backtest['Position_Change'] = backtest['Signal'].diff().abs()
    backtest['Return_Strategy'] -= backtest['Position_Change'] * TRANSACTION_COST
    backtest = backtest.dropna()
    backtest['Cumulative_Market']   = backtest['Return_Market'].cumsum().apply(np.exp)
    backtest['Cumulative_Strategy'] = backtest['Return_Strategy'].cumsum().apply(np.exp)

    total_return_strategy = backtest['Cumulative_Strategy'].iloc[-1] - 1
    total_return_market   = backtest['Cumulative_Market'].iloc[-1] - 1
    nb_trades             = int(backtest['Position_Change'].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Stratégie LSTM",  f"{total_return_strategy:.2%}")
    col2.metric("Buy & Hold",      f"{total_return_market:.2%}")
    col3.metric("Nombre de trades", nb_trades)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(backtest['Cumulative_Market'],   label='Buy & Hold', color='#4C72B0')
    ax.plot(backtest['Cumulative_Strategy'], label='Stratégie LSTM', color='#C44E52')
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax.set_title("Rendement cumulé")
    ax.set_ylabel("Rendement cumulé")
    ax.legend()
    st.pyplot(fig)
    plt.close()

# Onglet 4 - RÉSUMÉ

with tab4:
    st.subheader("Résumé du projet")

    st.markdown(f"""
    **Ticker analysé :** {ticker}  
    **Période :** {period} — intervalles de 5 minutes  
    **Bougies utilisées :** {len(df):,}  
    **Split :** 70% train / 15% validation / 15% test  

    ---

    ### Résultats des modèles
    | Modèle | Accuracy |
    |---|---|
    | Logistic Regression | {acc_lr:.2%} |
    | Random Forest | {acc_rf:.2%} |
    | **LSTM (seed=13, epochs={epochs})** | **{acc_lstm:.2%}** |

    ### Backtest
    | Stratégie | Rendement |
    |---|---|
    | LSTM | {total_return_strategy:.2%} |
    | Buy & Hold | {total_return_market:.2%} |
    | Nombre de trades | {nb_trades} |

    ---
    ### Limites & perspectives
    - Dataset limité à 60 jours — plus de données améliorerait le LSTM
    - Coûts de transaction simplifiés (0.01%)
    - Amélioration possible avec seuils de probabilité softmax
    - Extension possible à d'autres actifs ou timeframes
    - Seed et Epochs sont fixés pour la reproductibilité, la vitesse mais pourraient être optimisés.
    - Exploration de modèles plus complexes pour capturer des patterns temporels plus longs.
    ---
    ### Commentaire IA (Claude)
    """)
    
    if st.button("🤖 Générer un commentaire IA sur les résultats"):
        with st.spinner("Génération du commentaire..."):
            client = anthropic.Anthropic(api_key="") #J'ai pas mis ma clé car github ne l'autorise plus, mais vous pouvez la mettre ici pour que le code fonctionne.
            prompt = f"""Tu es un analyste quantitatif. Voici les résultats d'un backtest intraday sur {ticker} :
    - Stratégie LSTM : {total_return_strategy:.2%}
    - Buy & Hold : {total_return_market:.2%}
    - Nombre de trades : {nb_trades}
    - Accuracy LSTM : {acc_lstm:.2%}
    Rédige un commentaire professionnel de 3-4 phrases sur ces résultats."""
            message = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            st.session_state.commentaire_llm = message.content[0].text

    if st.session_state.get('commentaire_llm'):
        st.info(st.session_state.commentaire_llm)
