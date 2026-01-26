# analysis_plots.py
import pandas as pd
import plotly.express as px

def plot_price(data):
    price_col = data[["price", "label"]].copy()
    price_col["price"] = pd.to_numeric(price_col["price"], errors="coerce")
    price_col = price_col.dropna(subset=["price"])
    return px.histogram(price_col, x="price", color="label", nbins=60, barmode="overlay", opacity=0.6,
                        title="Distribution des prix (par label)")

def plot_top_categories(data, n=15):
    product = data["product_category_name_english"].astype(str).value_counts().head(n).reset_index()
    product.columns = ["category", "count"]
    return px.bar(product, x="category", y="count", title=f"Top {n} catégories produits")
    

def plot_top_cities(data, n=15):
    city = data["customer_city"].astype(str).value_counts().head(n).reset_index()
    city.columns = ["city", "count"]
    return px.bar(city, x="city", y="count", title=f"Répartition des clients par ville (Top {n})")

import plotly.express as px
import pandas as pd

def plot_length_vs_price(data):
    intermediate = data[["product_length_cm", "price"]].dropna().copy()
    intermediate["product_length_cm"] = pd.to_numeric(intermediate["product_length_cm"], errors="coerce")
    intermediate["price"] = pd.to_numeric(intermediate["price"], errors="coerce")
    final = intermediate.dropna()

    return px.scatter(
        final,
        x="product_length_cm",
        y="price",
        opacity=0.3,
        title="Relation entre longueur du produit et prix",
        labels={
            "product_length_cm": "Longueur du produit (cm)",
            "price": "Prix"
        }
    )
