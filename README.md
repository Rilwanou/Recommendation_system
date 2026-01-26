# 🚀 Recommendation System – FastAPI & Streamlit  

**Architecture découplée • Modèle ML centralisé • Dépendances gérées avec uv**

---

## 🧠 Objectif

Ce projet implémente un **système de recommandation simple et interprétable** :

- Attribue un **score de pertinence** à chaque paire (utilisateur, produit)  
- Le score représente la **probabilité d’achat**, basé sur :  
  - Caractéristiques du produit  
  - Contexte utilisateur  
  - Signaux dérivés de l’historique d’achats

Le système exploite l’idée que **les produits achetés ensemble ou par des clients similaires** sont plus susceptibles d’être achetés à nouveau.

---

## 🗂 Dataset

Olist dataset :

- Customers  
- Orders (livrés uniquement)  
- Order items  
- Products  
- Category name translation

---

## 🔍 Feature Engineering

Chaque interaction (user, product) est décrite par :

### Numériques
- price  
- product_photos_qty  
- product_weight_g  
- product_length_cm  
- product_height_cm  
- product_width_cm  
- purchase_count (popularité globale)  
- purchase_count_state (popularité locale par État)  
- recency_days (récence de l’achat)

### Catégorielles
- customer_state  
- customer_city  
- product_category_name_english

### Signaux relationnels
- cooc_score (co-occurrence produit–produit dans le même panier)

---

## 🎯 Labels (Data Mining)

- 1 → Interaction observée (achat)  
- 0 → Interaction non observée (negative sampling)  
- Ratio positif/négatif = 1:1  

> C’est un problème de **ranking / implicit-feedback**, pas une classification classique.

---

## ⚙ Preprocessing

- Numériques : median imputation + standardization  
- Catégorielles : imputation "missing" + OneHotEncoder

---

## 🧮 Modèle ML

- Logistic Regression avec régularisation L2 (Ridge)  
- Sélection de C via cross-validation  
- Modèle final : C = 0.05  
- Metrics : ROC-AUC, PR-AUC (Average Precision)

---

## 🏗 Architecture du projet
```
      project/
      ├── backend/
      │ ├── src/
      │ │ ├── api/
      │ │ │ ├── main.py # Point d’entrée FastAPI
      │ │ │ └── router.py # Routes API (v1)
      │ │ ├── services/
      │ │ │ ├── ingestion.py # Pipeline ingestion
      │ │ │ └── schemas.py # Schémas Pydantic
      │ │ └── init.py
      │ └── pyproject.toml # Dépendances backend (uv)
      │
      ├── frontend/
      │ ├── interface/
      │ │ ├── app.py # Application Streamlit
      │ │ └── api_client.py # Client HTTP
      │ └── pyproject.toml # Dépendances frontend (uv)
      │
      ├── models/
      │ ├── load_data.py # Chargement dataset
      │ ├── load_model.py # Chargement modèle ML
      │ ├── recommender.py # Algorithme recommandations
      │ └── logistic_ridge.joblib # Modèle entraîné
      │
      ├── data/
      │ └── processed/
      │ └── final_dataset.parquet
      │
      ├── notebooks/
      │ └── exploration.ipynb # Exploration data & features
      │
      └── README.md
```
---

## 🔄 Flux global

Dans l’interface Streamlit, l’utilisateur sélectionne un client existant.

Le frontend envoie la requête HTTP au backend (FastAPI).

Le backend récupère automatiquement :

Toutes les variables/features nécessaires pour le modèle depuis le dataset

Les produits candidats pour ce client

Le modèle calcule la probabilité d’achat (score) pour chaque produit candidat.

Les produits sont classés par score décroissant.

Le backend retourne le JSON des recommandations.

Le frontend affiche la liste des produits recommandés à l’utilisateur.

✅ L’utilisateur n’a pas à manipuler les features ou les produits : tout est généré automatiquement à partir du client sélectionné.


---


## 📦 Installation et lancement (backend + frontend)

### 1️⃣ Cloner le projet

```bash
git clone git@github.com:ril-hub46/Recommendation_system.git
cd Recommendation_system
uv sync
```


### 2️⃣ Lancer le backend
```
uvicorn backend.src.api.main:app --reload
```
API : http://localhost:8000
Swagger : http://localhost:8000/docs


### 3️⃣ Lancer le frontend
```
uv run streamlit run frontend/app.py
```
Interface : http://localhost:8501

Utilisateur
   ↓
Frontend (Streamlit)
   ↓ Requêtes HTTP
Backend (FastAPI)
   ↓
Chargement modèle & données (cache)
   ↓
Calcul des scores
   ↓
Réponse JSON
   ↑
Frontend (visualisation)

### 4️⃣ Pour entrainer le modèle
```bash
uv run python backend/train_logit.py
```





