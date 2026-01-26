"""
models/load_data.py

Fonctions de chargement et traitement des données
Utilisées par le backend - pas de dépendances Streamlit
"""

from pathlib import Path
import pandas as pd


def load_final_dataset():
    """Charge le dataset final depuis data/processed/final_dataset.parquet"""
    base_dir = Path(__file__).resolve().parents[1]  # Remonte à la racine
    data_path = base_dir / "data" / "processed" / "final_dataset.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset non trouvé : {data_path}")

    df = pd.read_parquet(data_path)
    return df


def get_unique_customers(df):
    """Retourne la liste unique des customer_unique_id"""
    if "customer_unique_id" in df.columns:
        return sorted(df["customer_unique_id"].dropna().unique().tolist())
    return []


def get_customer_profile(df, customer_unique_id):
    """
    Récupère le profil complet d'un client.
    Retourne les caractéristiques moyennes/principales du client.
    """
    customer_data = df[df["customer_unique_id"] == customer_unique_id]
    if customer_data.empty:
        return None

    # Colonnes du profil client (pas de produit)
    customer_cols = [
        "customer_unique_id",
        "customer_state",
        "customer_city",
        "purchase_count",
        "purchase_count_state",
        "cooc_score",
        "recency_days",
    ]

    # On prend la ligne la plus récente (ou première) pour le profil
    profile = customer_data.iloc[0][
        [col for col in customer_cols if col in customer_data.columns]
    ].to_dict()

    return profile


def get_all_products(df):
    """
    Retourne tous les produits uniques disponibles.
    Un produit = une combinaison unique de product_id + caractéristiques produit
    """
    # Colonnes produit
    product_cols = [
        "product_id",
        "product_category_name_english",
        "price",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    # Garder seulement les colonnes qui existent
    available_cols = [col for col in product_cols if col in df.columns]

    # Dédupliquer par product_id
    if "product_id" in df.columns:
        products = df[available_cols].drop_duplicates(subset=["product_id"]).copy()
    else:
        products = df[available_cols].drop_duplicates().copy()

    return products.reset_index(drop=True)


def get_customer_history(df, customer_unique_id):
    """
    Retourne l'historique d'achats d'un client.
    """
    customer_orders = df[df["customer_unique_id"] == customer_unique_id].copy()

    # Colonnes intéressantes pour l'historique
    hist_cols = [
        "product_id",
        "product_category_name_english",
        "price",
        "recency_days",
        "label",
    ]
    hist_cols = [col for col in hist_cols if col in customer_orders.columns]

    # Trier par recency_days (plus récent = valeur plus petite)
    if "recency_days" in customer_orders.columns:
        return customer_orders[hist_cols].sort_values("recency_days", ascending=True)
    else:
        return customer_orders[hist_cols]
