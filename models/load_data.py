"""
models/load_data.py

Data loading and processing functions
Used by the backend – no Streamlit dependencies
"""

from pathlib import Path
import pandas as pd


def load_final_dataset():
    """Load the final dataset from data/processed/final_dataset.parquet"""
    base_dir = Path(__file__).resolve().parents[1]  # Remonte à la racine
    data_path = base_dir / "data" / "processed" / "final_dataset.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_parquet(data_path)
    return df


def get_unique_customers(df):
    """Returns the unique list of customer_unique_id"""
    if "customer_unique_id" in df.columns:
        return sorted(df["customer_unique_id"].dropna().unique().tolist())
    return []


def get_customer_profile(df, customer_unique_id):
    """
    Retrieves the complete profile of a customer.
    Returns the average/main characteristics of the customer.
    """
    customer_data = df[df["customer_unique_id"] == customer_unique_id]
    if customer_data.empty:
        return None

    # Customer profile columns (not product)
    customer_cols = [
        "customer_unique_id",
        "customer_state",
        "customer_city",
        "purchase_count",
        "purchase_count_state",
        "cooc_score",
        "recency_days",
    ]

    # We take the most recent (or first) line for the profile.
    profile = customer_data.iloc[0][
        [col for col in customer_cols if col in customer_data.columns]
    ].to_dict()

    return profile


def get_all_products(df):
    """
    Returns all unique products available.
    A product = a unique combination of product_id + product characteristics
    """
    # Product columns
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

    # Keep only the columns that exist
    available_cols = [col for col in product_cols if col in df.columns]

    # Deduplicate by product_id
    if "product_id" in df.columns:
        products = df[available_cols].drop_duplicates(subset=["product_id"]).copy()
    else:
        products = df[available_cols].drop_duplicates().copy()

    return products.reset_index(drop=True)


def get_customer_history(df, customer_unique_id):
    """
    Returns the purchase history of a customer.
    """
    customer_orders = df[df["customer_unique_id"] == customer_unique_id].copy()

    # Interesting columns for the history
    hist_cols = [
        "product_id",
        "product_category_name_english",
        "price",
        "recency_days",
        "label",
    ]
    hist_cols = [col for col in hist_cols if col in customer_orders.columns]

    # Sort by recency_days (most recent = smallest value)
    if "recency_days" in customer_orders.columns:
        return customer_orders[hist_cols].sort_values("recency_days", ascending=True)
    else:
        return customer_orders[hist_cols]
