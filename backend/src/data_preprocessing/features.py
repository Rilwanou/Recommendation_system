import pandas as pd
from itertools import combinations
from collections import Counter


def add_recency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )
    reference_date = df["order_purchase_timestamp"].max()
    df["recency_days"] = (
        reference_date - df["order_purchase_timestamp"]
    ).dt.days
    return df


def add_item_popularity(df: pd.DataFrame) -> pd.DataFrame:
    popularity = (
        df.groupby("product_id")
        .size()
        .rename("purchase_count")
        .reset_index()
    )
    return df.merge(popularity, on="product_id", how="left")


def add_item_popularity_state(df: pd.DataFrame) -> pd.DataFrame:
    pop_state = (
        df.groupby(["customer_state", "product_id"])
        .size()
        .rename("purchase_count_state")
        .reset_index()
    )
    return df.merge(
        pop_state,
        on=["customer_state", "product_id"],
        how="left"
    )


def compute_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    items_per_order = (
        df.groupby("order_id")["product_id"].apply(list)
    )

    counter = Counter()
    for items in items_per_order:
        for pair in combinations(sorted(set(items)), 2):
            counter[pair] += 1

    cooc = pd.DataFrame(
        [(i, j, c) for (i, j), c in counter.items()],
        columns=["item_i", "item_j", "cooc_score"]
    )
    return cooc
