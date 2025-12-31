import numpy as np
import pandas as pd


def generate_negative_samples(
    df: pd.DataFrame,
    neg_ratio: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Génère des interactions négatives (user, item non acheté).
    """
    rng = np.random.default_rng(random_state)

    positives = df.copy()
    positives["label"] = 1
    # Initialisation cooc_score pour les interactions positives
    if "cooc_score" not in positives.columns:
        positives["cooc_score"] = 0

    all_items = positives["product_id"].unique()

    # Historique d'achat par user
    user_bought = (
        positives.groupby("customer_unique_id")["product_id"].apply(set).to_dict()
    )

    users = positives["customer_unique_id"].values
    n_neg = len(positives) * neg_ratio

    neg_rows = []

    for _ in range(n_neg):
        u = users[rng.integers(0, len(users))]
        bought = user_bought[u]

        # Tirage d'un item non acheté
        while True:
            it = all_items[rng.integers(0, len(all_items))]
            if it not in bought:
                break

        neg_rows.append((u, it))

    negatives_key = pd.DataFrame(neg_rows, columns=["customer_unique_id", "product_id"])

    # Métadonnées item (dédoublonnées)
    item_meta = positives[
        [
            "product_id",
            "product_category_name_english",
            "price",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
            "purchase_count",
        ]
    ].drop_duplicates("product_id")

    # Contexte user (state / city le plus fréquent)
    user_meta = (
        positives.groupby("customer_unique_id")[["customer_state", "customer_city"]]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
    )

    negatives = negatives_key.merge(
        user_meta, on="customer_unique_id", how="left"
    ).merge(item_meta, on="product_id", how="left")

    # Popularité locale
    if "purchase_count_state" in positives.columns:
        negatives = negatives.merge(
            positives[
                ["customer_state", "product_id", "purchase_count_state"]
            ].drop_duplicates(),
            on=["customer_state", "product_id"],
            how="left",
        )

    # Valeurs neutres
    negatives["cooc_score"] = 0
    negatives["recency_days"] = positives["recency_days"].median()
    negatives["label"] = 0

    # Colonnes nécessaires au modèle uniquement
    model_cols = [
        "customer_unique_id",
        "product_id",
        "customer_state",
        "customer_city",
        "product_category_name_english",
        "price",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "purchase_count",
        "purchase_count_state",
        "cooc_score",
        "recency_days",
        "label",
    ]

    positives = positives[model_cols]
    negatives = negatives[model_cols]

    return pd.concat([positives, negatives], ignore_index=True)
