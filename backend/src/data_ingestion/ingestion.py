from __future__ import annotations
from pathlib import Path
import warnings
import pandas as pd
from pydantic import BaseModel, Field, HttpUrl
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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

# Pydantic model to validate type hints before processing data


class SaveData(BaseModel):
    raw_dir: Path = Field(...)  # Directory is required
    filename: str = Field(..., min_length=1)


class CsvSource(BaseModel):
    url: HttpUrl | None = Field(
        default=None
    )  # URL is optional, defaults to None if not provided
    path: Path | None = Field(default=None)  # Local path

    def model_post_init(self, __context) -> None:
        # Validate that exactly one source is provided
        if (self.url is None) == (self.path is None):
            raise ValueError("Either 'url' or 'path' must be provided, but not both.")

        # Ensure the local file exists if a path is given
        if self.path is not None and not self.path.exists():
            raise FileNotFoundError(
                f"The specified CSV file does not exist: {self.path}"
            )


def load_csv(source: CsvSource) -> pd.DataFrame:
    if source.url is not None:
        logger.info(f"loading from {source.url}")
        return pd.read_csv(str(source.url))
    logger.info(f"loading from {source.path}")
    return pd.read_csv(source.path)


def save_csv(df: pd.DataFrame | None, filename: str, raw_dir: Path) -> None:
    params = SaveData(raw_dir=raw_dir, filename=filename)

    if df is None:
        print(f"Skipped {filename}")
        logger.info(f"Skipped {params.filename} (df is None)")
        return

    params.raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = params.raw_dir / params.filename
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")


def safe_len(df: pd.DataFrame | None) -> int:
    return 0 if df is None else len(df)


def default_raw_dir() -> Path:
    script_path = Path(__file__).resolve()
    backend_dir = script_path.parents[2]
    project_root = backend_dir.parent
    return project_root / "data" / "raw"


def run_ingestion(raw_dir: Path | None = None) -> None:
    raw_dir = raw_dir or default_raw_dir()

    logger.info("RAW_DIR :", raw_dir)

    df_customers = df_orders = df_order_items = df_products = None
    df_reviews = df_sellers = df_geolocation = df_payments = df_prod_translation = None

    try:
        print(" Olist data processing...")

        df_customers = load_csv(CsvSource(url=files_urls["customers"]))
        df_orders = load_csv(CsvSource(url=files_urls["orders"]))
        df_order_items = load_csv(CsvSource(url=files_urls["order_items"]))
        df_products = load_csv(CsvSource(url=files_urls["products"]))
        df_reviews = load_csv(CsvSource(url=files_urls["reviews"]))
        df_sellers = load_csv(CsvSource(url=files_urls["sellers"]))
        df_geolocation = load_csv(CsvSource(url=files_urls["geolocation"]))
        df_payments = load_csv(CsvSource(url=files_urls["payments"]))
        df_prod_translation = load_csv(CsvSource(url=files_urls["prod__english_name"]))

        logger.info(" Olist data loaded successfully !")

    except Exception as e:
        logger.info(" Failed to load real data.")
        logger.info(" DataFrames are None; skipping save automatically.")
        logger.info(f" Error details: {e}")

    print("\n datasets summary")
    print(f"   Customers              : {safe_len(df_customers):,}")
    print(f"   Orders                 : {safe_len(df_orders):,}")
    print(f"   Order items            : {safe_len(df_order_items):,}")
    print(f"   Products               : {safe_len(df_products):,}")
    print(f"   Reviews                : {safe_len(df_reviews):,}")
    print(f"   Sellers                : {safe_len(df_sellers):,}")
    print(f"   Geolocation            : {safe_len(df_geolocation):,}")
    print(f"   Payments               : {safe_len(df_payments):,}")
    print(f"   Category translation   : {safe_len(df_prod_translation):,}")

    save_csv(df_customers, "olist_customers_dataset.csv", raw_dir)
    save_csv(df_orders, "olist_orders_dataset.csv", raw_dir)
    save_csv(df_order_items, "olist_order_items_dataset.csv", raw_dir)
    save_csv(df_products, "olist_products_dataset.csv", raw_dir)
    save_csv(df_reviews, "olist_order_reviews_dataset.csv", raw_dir)
    save_csv(df_sellers, "olist_sellers_dataset.csv", raw_dir)
    save_csv(df_geolocation, "olist_geolocation_dataset.csv", raw_dir)
    save_csv(df_payments, "olist_order_payments_dataset.csv", raw_dir)
    save_csv(df_prod_translation, "product_category_name_translation.csv", raw_dir)


def run_ingestion_with_summary(raw_dir: Path | None = None) -> dict:
    raw_dir = raw_dir or default_raw_dir()
    run_ingestion(raw_dir=raw_dir)
    return {"raw_dir": str(raw_dir), "status": "done"}
