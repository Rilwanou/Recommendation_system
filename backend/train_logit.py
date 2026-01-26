from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from src.services.ingestion import load_data_raw
from src.data_preprocessing.merge import build_base_table
from src.data_preprocessing.features import (
    add_recency,
    add_item_popularity,
    add_item_popularity_state,
)
from src.data_preprocessing.negative_sampling import generate_negative_samples
from src.models.preprocessing import build_preprocessor
from src.models.train import train_model
from src.models.save_load import save_model


# ===============================
# PATHS
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_PATH = BASE_DIR / "models" / "logistic_ridge.joblib"


def main():
    print("BACKEND MAIN STARTED")

    # ===============================
    # DATA INGESTION + MERGE
    # ===============================
    dfs = load_data_raw(DATA_DIR)
    df = build_base_table(dfs)

    # ===============================
    # FEATURE ENGINEERING
    # ===============================
    df = add_recency(df)
    df = add_item_popularity(df)
    df = add_item_popularity_state(df)

    # ===============================
    # FEATURES
    # ===============================
    num_features = [
        "price",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "purchase_count",
        "purchase_count_state",
        "recency_days",
    ]

    cat_features = [
        "customer_state",
        "customer_city",
        "product_category_name_english",
    ]

    # ===============================
    # LABELS (positifs + négatifs)
    # ===============================
    df = generate_negative_samples(df, neg_ratio=1)

    X = df[num_features + cat_features]
    y = df["label"]

    print("\nDistribution des labels :")
    print(y.value_counts(normalize=True))

    # ===============================
    # TRAIN / TEST SPLIT
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ===============================
    # TRAIN MODEL
    # ===============================
    preprocessor = build_preprocessor(num_features, cat_features)
    model = train_model(preprocessor, X_train, y_train)

    # ===============================
    # EVALUATION
    # ===============================
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print("\n📊 Model performance on test set:")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")

    # ===============================
    # SAVE MODEL
    # ===============================
    save_model(model, MODEL_PATH)
    print(f"\n✅ Modèle entraîné et sauvegardé dans : {MODEL_PATH}")


if __name__ == "__main__":
    main()
