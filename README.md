# Recommendation System – Data Mining Project

## Objective
Propose a simple and interpretable recommendation system that assigns a relevance score to each (user, product) pair.
This score represents the probability that a user will purchase a product, based on product features, user context, and signals derived from purchase history.

The system relies on the idea that products bought together or by similar customers are more likely to be purchased again by other customers.

## Data
Olist dataset:
- customers
- orders (delivered only)
- order items
- products
- category name translation

## Feature Engineering
Each (user, product) interaction is described by:

### Numerical Features
- price
- product_photos_qty
- product_weight_g
- product_length_cm
- product_height_cm
- product_width_cm
- purchase_count (global product popularity)
- purchase_count_state (local popularity by state)
- recency_days (purchase recency)

### Categorical Features
- customer_state
- customer_city
- product_category_name_english

### Relational Signal
- cooc_score (product–product co-occurrence within the same basket)

## Labels (Data Mining)
- Label = 1: observed interaction (purchase)
- Label = 0: non-observed interaction (negative sampling)
Positive / negative ratio = 1:1.

This is an implicit-feedback recommendation problem (ranking), not a classical classification task.

## Preprocessing
- Numerical: median imputation + standardization
- Categorical: "missing" imputation + OneHotEncoder

## Model
- Logistic regression with L2 regularization (Ridge)
- C selection via cross-validation
- Final model: C = 0.05
- Metrics: ROC-AUC, PR-AUC (Average Precision)

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

## Run the FastAPI API
From the project root:
uvicorn frontend.api.main:app --reload

API: http://127.0.0.1:8000  
Docs: http://127.0.0.1:8000/docs

Role: provide a recommendation score for a user–product interaction.

## Run Streamlit
uv run streamlit run frontend/interface/app.py

Role: demonstration interface allowing users to input user/product information and display the score returned by the API.

## Global Logic
1. The user provides a set of candidate products (with their features).
2. Each product is transformed into numerical and categorical features identical to those used during training.
3. The model computes, for each product, a purchase probability (score).
4. Products are ranked by decreasing score to produce the recommendation.

## Component Roles
- Model (backend): computes a relevance score for each candidate product.
- FastAPI API: exposes the model through a scoring endpoint.
- Streamlit interface: allows testing the system by entering products (sample_data.py) and visualizing scores and rankings.

