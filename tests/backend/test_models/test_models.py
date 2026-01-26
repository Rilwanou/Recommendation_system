from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import pytest
import joblib

# =====================================
# Ajouter la racine du projet au path
# =====================================
root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(root))

# Imports pipeline
from backend.src.services.ingestion import load_data_raw
from backend.src.data_preprocessing.merge import build_base_table
from backend.src.data_preprocessing.features import (
    add_recency,
    add_item_popularity,
    add_item_popularity_state,
)
from backend.src.data_preprocessing.negative_sampling import generate_negative_samples
from backend.src.models.preprocessing import build_preprocessor
from backend.src.models.train import train_model
from backend.src.models.save_load import save_model



def _sample_df_raw(tmp_path: Path) -> Path:
    """Créer un dataset fictif suffisamment grand pour 5-fold CV"""
    data_dir = tmp_path / "data_raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    n = 10  # 10 clients × 10 produits → 100 lignes après merge
    # Customers
    pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "customer_unique_id": [f"u{i}" for i in range(n)],
        "customer_state": ["SP","RJ","MG","RS","BA","RJ","SP","MG","BA","RS"],
        "customer_city": [f"city{i}" for i in range(n)]
    }).to_csv(data_dir / "olist_customers_dataset.csv", index=False)

    # Orders
    pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "customer_id": [f"c{i}" for i in range(n)],
        "order_purchase_timestamp": ["2025-01-01 10:00:00"]*n
    }).to_csv(data_dir / "olist_orders_dataset.csv", index=False)

    # Order Items
    pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "product_id": [f"p{i}" for i in range(n)],
        "price": [10.0+i for i in range(n)]
    }).to_csv(data_dir / "olist_order_items_dataset.csv", index=False)

    # Products
    pd.DataFrame({
        "product_id": [f"p{i}" for i in range(n)],
        "product_category_name": [f"cat{i}" for i in range(n)],
        "product_weight_g": [100+i for i in range(n)],
        "product_length_cm": [10+i for i in range(n)],
        "product_height_cm": [5+i for i in range(n)],
        "product_width_cm": [2+i for i in range(n)],
        "product_photos_qty": [1 for _ in range(n)]
    }).to_csv(data_dir / "olist_products_dataset.csv", index=False)

    # Translation
    pd.DataFrame({
        "product_category_name": [f"cat{i}" for i in range(n)],
        "product_category_name_english": [f"cat_en{i}" for i in range(n)]
    }).to_csv(data_dir / "product_category_name_translation.csv", index=False)

    # Reviews
    pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "review_score": [5 for _ in range(n)]
    }).to_csv(data_dir / "olist_order_reviews_dataset.csv", index=False)

    # Sellers
    pd.DataFrame({
        "seller_id": [f"s{i}" for i in range(n)],
        "seller_zip_code_prefix": [1000+i for i in range(n)],
        "seller_city": [f"city{i}" for i in range(n)],
        "seller_state": ["SP","RJ","MG","RS","BA","RJ","SP","MG","BA","RS"]
    }).to_csv(data_dir / "olist_sellers_dataset.csv", index=False)

    # Payments
    pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "payment_sequential": [1]*n,
        "payment_type": ["credit_card"]*n,
        "payment_installments": [1]*n,
        "payment_value": [10.0+i for i in range(n)]
    }).to_csv(data_dir / "olist_order_payments_dataset.csv", index=False)

    # Geolocation (optionnel pour tests)
    pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "geolocation_lat": [0.0]*n,
        "geolocation_lng": [0.0]*n,
        "geolocation_zip_code_prefix": [1000+i for i in range(n)]
    }).to_csv(data_dir / "olist_geolocation_dataset.csv", index=False)

    return data_dir



# =====================================
# Tests
# =====================================

class TestIngestionAndMerge:
    def test_load_and_merge(self, tmp_path: Path):
        data_dir = _sample_df_raw(tmp_path)
        dfs = load_data_raw(data_dir)
        df = build_base_table(dfs)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # Vérification des colonnes après merge
        for col in ["customer_unique_id", "product_id", "price", "product_category_name_english"]:
            assert col in df.columns

class TestFeatureEngineering:
    def test_add_features(self, tmp_path: Path):
        data_dir = _sample_df_raw(tmp_path)
        df = build_base_table(load_data_raw(data_dir))
        
        df = add_recency(df)
        df = add_item_popularity(df)
        df = add_item_popularity_state(df)
        
        for col in ["recency_days", "purchase_count", "purchase_count_state"]:
            assert col in df.columns
            assert not df[col].isna().all()

class TestNegativeSampling:
    def test_generate_negative_samples(self, tmp_path: Path):
        data_dir = _sample_df_raw(tmp_path)
        df = build_base_table(load_data_raw(data_dir))
        df = add_recency(df)
        df = add_item_popularity(df)
        df = add_item_popularity_state(df)
        
        # On génère les labels négatifs
        df_ml = generate_negative_samples(df, neg_ratio=1)
        
        assert "label" in df_ml.columns
        assert 0 in df_ml["label"].values
        assert 1 in df_ml["label"].values

class TestFullPipeline:
    def test_full_main_flow(self, tmp_path: Path):
        """Pipeline complet du chargement à la sauvegarde"""
        data_dir = _sample_df_raw(tmp_path)
        
        # ETL & Feature Engineering
        df = build_base_table(load_data_raw(data_dir))
        df = add_recency(df)
        df = add_item_popularity(df)
        df = add_item_popularity_state(df)
        df = generate_negative_samples(df, neg_ratio=1)
        
        num_features = ["price","product_photos_qty","product_weight_g","product_length_cm",
                        "product_height_cm","product_width_cm","purchase_count","purchase_count_state",
                        "recency_days"]
        cat_features = ["customer_state","customer_city","product_category_name_english"]
        
        X = df[num_features + cat_features]
        y = df["label"]
        
        # Training
        preprocessor = build_preprocessor(num_features, cat_features)
        model = train_model(preprocessor, X, y)
        
        # Sauvegarde
        model_path = tmp_path / "model.joblib"
        save_model(model, model_path)
        
        assert model_path.exists()
        loaded = joblib.load(model_path)
        assert hasattr(loaded, "predict_proba")