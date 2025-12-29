import pandas as pd
from pathlib import Path


def load_raw_data(data_dir: Path) -> dict:
    """
    Charge toutes les tables CSV brutes.
    """
    return {
        "customers": pd.read_csv(data_dir / "olist_customers_dataset.csv"),
        "orders": pd.read_csv(data_dir / "olist_orders_dataset.csv"),
        "order_items": pd.read_csv(data_dir / "olist_order_items_dataset.csv"),
        "products": pd.read_csv(data_dir / "olist_products_dataset.csv"),
        "category_translation": pd.read_csv(
            data_dir / "product_category_name_translation.csv"
        ),
    }
