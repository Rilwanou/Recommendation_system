import pandas as pd
from pathlib import Path
from src.services.ingestion import load_data_raw
from src.data_preprocessing.merge import build_base_table
from src.data_preprocessing.labels import create_positive_labels
from src.data_preprocessing.features import (
    add_recency, 
    add_item_popularity, 
    add_item_popularity_state
)

from src.data_preprocessing.negative_sampling import generate_negative_samples
from pathlib import Path



# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.parquet"

def run_full_preprocessing():
    print("🚀 Démarrage du pipeline de traitement...")

    # 1. Chargement
    dfs = load_data_raw(RAW_DATA_DIR)
    
    # 2. Merge & Nettoyage de base
    df = build_base_table(dfs)
    df = create_positive_labels(df)
    
    # 3. Feature Engineering
    print("⚙️ Calcul des features (Récence, Popularité...)...")
    df = add_recency(df)
    df = add_item_popularity(df)
    df = add_item_popularity_state(df)
    
    # 4. Negative Sampling (Transformation pour le ML)
    print("🧪 Génération des échantillons négatifs...")
    df_final = generate_negative_samples(df, neg_ratio=1)
    
    # 5. Sauvegarde
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(PROCESSED_DATA_PATH, index=False)
    
    print(f"✅ Données traitées sauvegardées ici : {PROCESSED_DATA_PATH}")
    print(f"📊 Taille finale du dataset : {df_final.shape}")

if __name__ == "__main__":
    run_full_preprocessing()