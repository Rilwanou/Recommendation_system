from __future__ import annotations

from pathlib import Path
import warnings
import pandas as pd
from pydantic import BaseModel, Field, HttpUrl

warnings.filterwarnings("ignore")

BASE_URL = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"

files_urls = {
    "customers": BASE_URL + "olist_customers_dataset.csv",
    "orders": BASE_URL + "olist_orders_dataset.csv",
    "order_items": BASE_URL + "olist_order_items_dataset.csv",
    "products": BASE_URL + "olist_products_dataset.csv",
    "reviews": BASE_URL + "olist_order_reviews_dataset.csv",
    "sellers": BASE_URL + "olist_sellers_dataset.csv",
    "geolocation": BASE_URL + "olist_geolocation_dataset.csv",
    "payments": BASE_URL + "olist_order_payments_dataset.csv",
    "prod__english_name": BASE_URL + "product_category_name_translation.csv",
}


SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[2]      
PROJECT_ROOT = BACKEND_DIR.parent         
RAW_DIR = PROJECT_ROOT / "data" / "raw"   
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("📁 Project root :", PROJECT_ROOT)
print("📁 RAW_DIR      :", RAW_DIR)

class SaveCsvParams(BaseModel):
    """Validation model for CSV saving parameters."""
    raw_dir: Path = Field(..., description="Target directory where CSV files will be saved.")
    filename: str = Field(..., min_length=1, description="Output CSV filename (e.g., 'orders.csv').")


class CsvSource(BaseModel):
    """
    CSV source definition.

    Exactly one of `url` or `path` must be provided.
    """
    url: HttpUrl | None = Field(default=None, description="HTTP(S) URL to a CSV file.")
    path: Path | None = Field(default=None, description="Local path to a CSV file.")

    def model_post_init(self, __context) -> None:
        if (self.url is None) == (self.path is None):
            raise ValueError("Provide exactly one of `url` or `path` (not both).")

        if self.path is not None and not self.path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.path}")


def load_csv(source: CsvSource) -> pd.DataFrame:
    """
    Load a CSV from a validated source (URL or local path).
    """
    if source.url is not None:
        return pd.read_csv(str(source.url))
    return pd.read_csv(source.path)  


def save_csv(df: pd.DataFrame | None, filename: str, raw_dir: Path) -> None:
    """
    Save a DataFrame as a CSV file into raw_dir using validated Pydantic params.
    """
    params = SaveCsvParams(raw_dir=raw_dir, filename=filename)

    if df is None:
        print(f"⚠️ Skipped {params.filename} (df is None)")
        return

    params.raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = params.raw_dir / params.filename

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Saved: {out_path}")


def safe_len(df: pd.DataFrame | None) -> int:
    return 0 if df is None else len(df)

df_customers = df_orders = df_order_items = df_products = None
df_reviews = df_sellers = df_geolocation = df_payments = df_prod_translation = None

try:
    print("\n⬇️ Chargement des données réelles Olist...")

    df_customers = load_csv(CsvSource(url=files_urls["customers"]))
    df_orders = load_csv(CsvSource(url=files_urls["orders"]))
    df_order_items = load_csv(CsvSource(url=files_urls["order_items"]))
    df_products = load_csv(CsvSource(url=files_urls["products"]))
    df_reviews = load_csv(CsvSource(url=files_urls["reviews"]))
    df_sellers = load_csv(CsvSource(url=files_urls["sellers"]))
    df_geolocation = load_csv(CsvSource(url=files_urls["geolocation"]))
    df_payments = load_csv(CsvSource(url=files_urls["payments"]))
    df_prod_translation = load_csv(CsvSource(url=files_urls["prod__english_name"]))

    print("✅ Données Olist chargées avec succès !")

except Exception as e:
    print("⚠️ Impossible de charger les données réelles.")
    print("👉 Les DataFrames restent à None, donc sauvegarde skip automatiquement.")
    print(f"ℹ️ Raison : {e}")

print("\n📊 Résumé des datasets")
print(f"   👥 Customers              : {safe_len(df_customers):,}")
print(f"   📦 Orders                 : {safe_len(df_orders):,}")
print(f"   🛒 Order items            : {safe_len(df_order_items):,}")
print(f"   📱 Products               : {safe_len(df_products):,}")
print(f"   ⭐ Reviews                : {safe_len(df_reviews):,}")
print(f"   🏪 Sellers                : {safe_len(df_sellers):,}")
print(f"   📍 Geolocation            : {safe_len(df_geolocation):,}")
print(f"   💳 Payments               : {safe_len(df_payments):,}")
print(f"   🌐 Category translation   : {safe_len(df_prod_translation):,}")

save_csv(df_customers, "olist_customers_dataset.csv", RAW_DIR)
save_csv(df_orders, "olist_orders_dataset.csv", RAW_DIR)
save_csv(df_order_items, "olist_order_items_dataset.csv", RAW_DIR)
save_csv(df_products, "olist_products_dataset.csv", RAW_DIR)
save_csv(df_reviews, "olist_order_reviews_dataset.csv", RAW_DIR)
save_csv(df_sellers, "olist_sellers_dataset.csv", RAW_DIR)
save_csv(df_geolocation, "olist_geolocation_dataset.csv", RAW_DIR)
save_csv(df_payments, "olist_order_payments_dataset.csv", RAW_DIR)
save_csv(df_prod_translation, "product_category_name_translation.csv", RAW_DIR)
