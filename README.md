# Prédiction de prix intraday — AAPL

Projet réalisé dans le cadre du cours GFN 260 (Machine Learning en Assurance et Finance)
CNAM 2026
Yannis Khalfi & Branko Markovic

## Description

On cherche à prédire si le prix d'AAPL va monter ou descendre dans les 5 prochaines minutes,
à partir de données haute fréquence récupérées via Yahoo Finance.

On a d'abord travaillé sur des données LVMH haute fréquence (Euronext BEDOFIH) mais le manque
de données suffisantes nous a amenés à basculer sur AAPL via Yahoo Finance.

On compare trois approches : régression logistique, random forest, et un LSTM PyTorch.
On backteste ensuite la meilleure stratégie sur les données de test.

## Structure du projet

- `1_LVMH_exploration/` — exploration initiale sur données LVMH haute fréquence
- `2_AAPL_pipeline/` — pipeline complet AAPL (features, modèles, backtest, streamlit)
- `report/` — rapport PDF du projet

## Installation

pip install -r requirements.txt

## Lancer le dashboard

streamlit run 2_AAPL_pipeline/streamlit_app.py

## Données

Récupérées automatiquement via yfinance au lancement du script (AAPL, 60 jours, 5 minutes).
Les données LVMH proviennent de la base Euronext BEDOFIH (accès restreint).

## NB

La clé API pour Anthropic a été volontairement retiré suite aux restrictions de github pour 
push le code. Vous pouvez ajouter la votre pour faire fonctionner l'IA Gen.
