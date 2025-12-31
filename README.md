# Recommendation System – Data Mining Project

## Objectif
Proposer un système de recommandation simple et interprétable qui attribue à chaque couple (utilisateur, produit) un score de pertinence.
Ce score représente la probabilité que l’utilisateur achète le produit, à partir des caractéristiques du produit, du contexte utilisateur et de signaux issus de l’historique d’achats.

Le système s’appuie sur le fait que des produits achetés ensemble ou par des clients similaires ont de fortes chances d’être achetés à nouveau par d’autres clients.

## Données
Données Olist :
- clients
- commandes (livrées uniquement)
- items de commande
- produits
- traduction des catégories

## Feature engineering
Chaque interaction (utilisateur, produit) est décrite par :

### Variables numériques
- price
- product_photos_qty
- product_weight_g
- product_length_cm
- product_height_cm
- product_width_cm
- purchase_count (popularité globale du produit)
- purchase_count_state (popularité locale par État)
- recency_days (récence de l’achat)

### Variables catégorielles
- customer_state
- customer_city
- product_category_name_english

### Signal relationnel
- cooc_score (co-occurrence produit–produit dans un même panier)

## Labels (data mining)
- Label = 1 : interaction observée (achat)
- Label = 0 : interaction non observée (negative sampling)
Ratio positifs / négatifs = 1:1.

Il s’agit d’un problème de recommandation à feedback implicite (ranking), et non d’une classification classique.

## Prétraitement
- Numérique : imputation médiane + standardisation
- Catégoriel : imputation "missing" + OneHotEncoder

## Modèle
- Régression logistique avec régularisation L2 (Ridge)
- Sélection de C par validation croisée
- Modèle final : C = 0.05
- Métriques : ROC-AUC, PR-AUC (Average Precision)

## Architecture
backend/
  src/
    data_ingestion/
    data_preprocessing/
    modeling/

frontend/
  app.py          # Streamlit
  api/            # FastAPI

models/
  logistic_ridge.joblib

notebooks/
  exploration.ipynb

## Lancer l’API FastAPI
Depuis la racine du projet :
uvicorn frontend.api.main:app --reload

API : http://127.0.0.1:8000
Docs : http://127.0.0.1:8000/docs

Rôle : fournir un score de recommandation pour une interaction utilisateur–produit.

## Lancer Streamlit
python -m streamlit run frontend/app.py

Rôle : interface de démonstration permettant de saisir des informations utilisateur / produit et d’afficher le score retourné par l’API.

## Logique globale
1. L’utilisateur fournit un ensemble de produits candidats (avec leurs caractéristiques).
3. Chaque produit est transformé en features numériques et catégorielles, identiques à celles utilisées à l’entraînement.
4. Le modèle calcule, pour chaque produit, une probabilité d’achat (score).
5. Les produits sont triés par score décroissant pour produire une recommandation

## Role des composantes

- Modèle (backend) : calcule un score de pertinence pour chaque produit candidat.

- API FastAPI : expose le modèle via un endpoint de scoring.

- Interface Streamlit : permet de tester le système en saisissant des produits (sample_data.py) et en visualisant les scores et le classement.
