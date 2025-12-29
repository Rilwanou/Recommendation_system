import pandas as pd


def create_positive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la variable cible implicite :
    label = 1 pour les interactions observées (achat).
    """
    df = df.copy()
    df["label"] = 1
    return df
