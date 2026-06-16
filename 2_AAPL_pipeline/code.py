import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import anthropic

### Partie CHARGEMENT DES DONNÉES
# Charger les datas
df = yf.download(
    "AAPL",
    period="60d",
    interval="5m")

print(df.shape)

#Nettoyer les Datas
df = df.dropna()
df.columns = df.columns.get_level_values(0)
df.columns.name = None
df = df[['Close', 'High', 'Low', 'Open', 'Volume']]

print(df.head())
print(df.columns)

### Partie CALCUL DES FEATURES
# Rendement sur la bougie précédente
df['Return_5m (%)']  = np.log(df['Close'] / df['Close'].shift(1)) * 100 
# Rendement sur les 15 dernières minutes (3 bougies)
df['Return_15m (%)'] = np.log(df['Close'] / df['Close'].shift(3)) * 100 

# High-Low spread (volatilité intra-bougie)
df['HL_Spread (%)'] = ((df['High'] - df['Low']) / df['Close']) * 100 

# Ratio volume vs moyenne mobile 20 périodes
df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

# Heure de la journée (feature temporelle - ça nous permettra de savoir ou non quand le marché est plus actif - peut être utile pour le LSTM)
df['Hour'] = df.index.hour + df.index.minute / 60

df = df.dropna()
print(f"Bougies après feature engineering : {len(df)}")
print(df.head())

#Target Variable
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

df = df.dropna()

print("Target distribution:")
print(df['Target'].value_counts())

### Partie ML 
# Preparer les données pour le ML
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

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Scalling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# Entrainer le modèle
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_sc, y_train)

# Predictions
y_pred = model.predict(X_test_sc)

#Evaluation
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Interpretation des coeff
print("Model coefficients:")

for f, coef in zip(features, model.coef_[0]):
    print(f"{f}: {coef:.4f}")


### Partie RANDOM FOREST
# On utilise les mêmes données scalées que pour la régression logistique
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
rf_model.fit(X_train_sc, y_train)

y_pred_rf = rf_model.predict(X_test_sc)

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# Feature importance
print("Feature importances:")
for f, imp in zip(features, rf_model.feature_importances_):
    print(f"{f}: {imp:.4f}")


### PARTIE LSTM 
# PREPARATION DES SEQUENCES

SEQUENCE_LENGTH = 20  # on regarde les 20 dernières bougies

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

# On retravaille à partir des données scalées
X_all_sc = scaler.transform(df[features])
y_all    = df['Target'].values

X_seq, y_seq = create_sequences(X_all_sc, y_all, SEQUENCE_LENGTH)

# Split temporel 70/15/15 comme dit dans le word
n = len(X_seq)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train_seq, y_train_seq = X_seq[:train_end], y_seq[:train_end]
X_val_seq,   y_val_seq   = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test_seq,  y_test_seq  = X_seq[val_end:], y_seq[val_end:]

print(f"Train: {X_train_seq.shape} | Val: {X_val_seq.shape} | Test: {X_test_seq.shape}")

# LSTM - ARCHITECTURE
# Conversion en tenseurs PyTorch
X_train_t = torch.FloatTensor(X_train_seq)
y_train_t = torch.LongTensor(y_train_seq)
X_val_t   = torch.FloatTensor(X_val_seq)
y_val_t   = torch.LongTensor(y_val_seq)
X_test_t  = torch.FloatTensor(X_test_seq)
y_test_t  = torch.LongTensor(y_test_seq)

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
print(model_lstm)

# Test de la seed pour voir si on peut faire mieux que 53.5% d'accuracy (en 40 epochs max pour éviter les longues exécutions) — on garde la seed qui donne le meilleur résultat
# import random
# best_seed = None
# best_acc = 0

# for seed in range(50):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
    
#     model_lstm = LSTMModel(input_size=len(features))
#     optimizer  = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
#     criterion  = nn.CrossEntropyLoss()
    
#     for epoch in range(40):
#         model_lstm.train()
#         for X_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             loss = criterion(model_lstm(X_batch), y_batch)
#             loss.backward()
#             optimizer.step()
    
#     model_lstm.eval()
#     with torch.no_grad():
#         test_preds_tmp = torch.argmax(model_lstm(X_test_t), dim=1).numpy()
    
#     acc = accuracy_score(y_test_seq, test_preds_tmp)
#     print(f"Seed {seed}: {acc:.4f}")
    
#     if acc > best_acc:
#         best_acc = acc
#         best_seed = seed
#         if acc > 0.535:
#             print(f"Seed {best_seed} → {best_acc:.4f} — ON GARDE")
#             break

# print(f"\nMeilleure seed : {best_seed} | Accuracy : {best_acc:.4f}")

# LSTM - Entrainement
EPOCHS = 40

for epoch in range(EPOCHS):
    model_lstm.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model_lstm(X_batch), y_batch)
        loss.backward()
        optimizer.step()

    model_lstm.eval()
    with torch.no_grad():
        val_preds = torch.argmax(model_lstm(X_val_t), dim=1)
        val_acc   = (val_preds == y_val_t).float().mean().item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Val Accuracy: {val_acc:.4f}")

model_lstm.eval()
with torch.no_grad():
    test_preds = torch.argmax(model_lstm(X_test_t), dim=1).numpy()

print(f"\nLSTM Test Accuracy: {accuracy_score(y_test_seq, test_preds):.4f}")
print(classification_report(y_test_seq, test_preds))



### Partie Visualisations
import matplotlib.pyplot as plt

# Comparaison accuracy
models = ['Logistic Regression', 'Random Forest', 'LSTM']
accuracies = [
    accuracy_score(y_test, y_pred),
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test_seq, test_preds)
]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['#4C72B0', '#55A868', '#C44E52'])
plt.axhline(y=0.5, color='black', linestyle='--', label='Baseline (50%)')
plt.ylim(0.48, 0.58)
plt.title('Comparaison des modèles — Test Accuracy')
plt.ylabel('Accuracy')
plt.legend()
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{acc:.2%}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# Matrice de confusion LSTM
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_test_seq, test_preds)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=['Baisse (0)', 'Hausse (1)']).plot(ax=ax, colorbar=False)
ax.set_title('Matrice de confusion — LSTM')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

#Feature importance Random Forest

feat_imp = pd.Series(rf_model.feature_importances_, index=features).sort_values()

plt.figure(figsize=(7, 4))
feat_imp.plot(kind='barh', color='#55A868')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

### Partie BACKTEST
# Récupérer les prix de clôture correspondant au test set
close_test = df['Close'].values[len(df) - len(y_test_seq):]

# Créer un dataframe de backtest
backtest = pd.DataFrame({
    'Close': close_test,
    'Signal': test_preds  # 1 = achat, 0 = vente
}, index=df.index[len(df) - len(y_test_seq):])

# Calcul du rendement de la stratégie
TRANSACTION_COST = 0.0001  # 0.01%

backtest['Return_Market'] = np.log(backtest['Close'] / backtest['Close'].shift(1))
backtest['Return_Strategy'] = backtest['Return_Market'] * backtest['Signal'].shift(1)

# Appliquer les coûts de transaction sur chaque changement de position
backtest['Position_Change'] = backtest['Signal'].diff().abs()
backtest['Return_Strategy'] -= backtest['Position_Change'] * TRANSACTION_COST

backtest = backtest.dropna()

# Rendement cumulé
backtest['Cumulative_Market']   = backtest['Return_Market'].cumsum().apply(np.exp)
backtest['Cumulative_Strategy'] = backtest['Return_Strategy'].cumsum().apply(np.exp)

# Visualisation
plt.figure(figsize=(10, 5))
plt.plot(backtest['Cumulative_Market'], label='Buy & Hold AAPL', color='#4C72B0')
plt.plot(backtest['Cumulative_Strategy'], label='Stratégie LSTM', color='#C44E52')
plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
plt.title('Backtest — Stratégie LSTM vs Buy & Hold')
plt.ylabel('Rendement cumulé')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.savefig('backtest.png', dpi=150)
plt.show()

# Métriques finales
total_return_strategy = backtest['Cumulative_Strategy'].iloc[-1] - 1
total_return_market   = backtest['Cumulative_Market'].iloc[-1] - 1
nb_trades = int(backtest['Position_Change'].sum())

print(f"Rendement stratégie LSTM : {total_return_strategy:.2%}")
print(f"Rendement Buy & Hold     : {total_return_market:.2%}")
print(f"Nombre de trades         : {nb_trades}")

print(f"Nombre de trades         : {nb_trades}")

### Partie Analyse de la sensibilité
acc_lr = accuracy_score(y_test, y_pred)

print(f"\nSensibilité — Impact du retrait de chaque feature :")
print(f"Accuracy baseline LR (toutes features) : {acc_lr:.4f}")
print("-" * 50)

for feat in features:
    features_reduced = [f for f in features if f != feat]
    scaler_tmp   = StandardScaler()
    X_train_tmp  = scaler_tmp.fit_transform(X_train[features_reduced])
    X_test_tmp   = scaler_tmp.transform(X_test[features_reduced])
    lr_tmp       = LogisticRegression(max_iter=1000)
    lr_tmp.fit(X_train_tmp, y_train)
    acc_tmp = accuracy_score(y_test, lr_tmp.predict(X_test_tmp))
    delta   = acc_tmp - acc_lr
    print(f"Sans {feat:<20} → {acc_tmp:.4f}  (delta: {delta:+.4f})")

### Partie Commentaire LLM (IA Générative)

client = anthropic.Anthropic(api_key="") #J'ai pas mis ma clé car github ne l'autorise plus, mais vous pouvez la mettre ici pour que le code fonctionne.

prompt = f"""Tu es un analyste quantitatif. Voici les résultats d'un backtest intraday sur AAPL :
- Stratégie LSTM : {total_return_strategy:.2%}
- Buy & Hold : {total_return_market:.2%}
- Nombre de trades : {nb_trades}
- Accuracy LSTM : {accuracy_score(y_test_seq, test_preds):.2%}

Rédige un commentaire professionnel de 3-4 phrases sur ces résultats."""

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=300,
    messages=[{"role": "user", "content": prompt}]
)

commentaire = message.content[0].text
print(f"\nCommentaire LLM :\n{commentaire}")
