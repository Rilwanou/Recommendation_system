"""
backend/src/api/router.py

Routes API pour le système de recommandation Olist
Utilise le dataset final_dataset.parquet
"""

import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

from services.ingestion import run_ingestion_with_summary
from services.schemas import ItemFeatures, PredictionResponse
from services.dependencies import get_model


# =====================================
# Router principal
# =====================================

router = APIRouter(tags=["recommendation_system"])


# =====================================
# Schémas Pydantic
# =====================================

class IngestionRequest(BaseModel):
    raw_subdir: str | None = None


class RecommendationRequest(BaseModel):
    customer_unique_id: str = Field(..., description="ID unique du client")
    n_recommendations: int = Field(10, ge=1, le=50, description="Nombre de recommandations")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Score minimum")


class RecommendationItem(BaseModel):
    rank: int
    product_id: str
    price: float
    purchase_probability: float


class RecommendationResponse(BaseModel):
    customer_unique_id: str
    total_recommendations: int
    generated_at: str
    recommendations: List[RecommendationItem]


class CustomerProfile(BaseModel):
    customer_unique_id: str
    customer_state: str
    customer_city: str
    purchase_count: int
    purchase_count_state: int
    cooc_score: float
    recency_days: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool
    timestamp: str


# =====================================
# Fonctions utilitaires - Données
# =====================================

def load_final_dataset() -> pd.DataFrame:
    """Charge le dataset final depuis data/processed/final_dataset.parquet"""
    base_dir = Path(__file__).resolve().parents[3]  # Remonte à la racine du projet
    data_path = base_dir / "data" / "processed" / "final_dataset.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset non trouvé : {data_path}")
    
    df = pd.read_parquet(data_path)
    return df


def get_unique_customers_list(df: pd.DataFrame) -> List[str]:
    """Retourne la liste unique des customer_unique_id"""
    if "customer_unique_id" in df.columns:
        return sorted(df["customer_unique_id"].dropna().unique().tolist())
    return []


def get_customer_profile_data(df: pd.DataFrame, customer_unique_id: str) -> dict:
    """Récupère le profil d'un client"""
    customer_data = df[df["customer_unique_id"] == customer_unique_id]
    if customer_data.empty:
        return None
    
    # Prendre la première ligne pour le profil
    profile = customer_data.iloc[0].to_dict()
    
    # Retirer les colonnes produit
    keys_to_remove = ['product_id', 'product_category_name_english', 'price',
                      'product_photos_qty', 'product_weight_g', 'product_length_cm',
                      'product_height_cm', 'product_width_cm', 'label']
    
    for key in keys_to_remove:
        profile.pop(key, None)
    
    return profile


def get_all_products_data(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne tous les produits uniques disponibles"""
    product_cols = [
        'product_id',
        'product_category_name_english',
        'price',
        'product_photos_qty',
        'product_weight_g',
        'product_length_cm',
        'product_height_cm',
        'product_width_cm'
    ]
    
    available_cols = [col for col in product_cols if col in df.columns]
    
    if 'product_id' in df.columns:
        products = df[available_cols].drop_duplicates(subset=['product_id']).copy()
    else:
        products = df[available_cols].drop_duplicates().copy()
    
    return products.reset_index(drop=True)


# =====================================
# Fonction de recommandation
# =====================================

def compute_recommendations(
    customer_profile: dict,
    products_df: pd.DataFrame,
    top_k: int = 10,
    min_score: float = 0.0
) -> pd.DataFrame:
    """
    Calcule les scores de recommandation pour les produits candidats.
    """
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Modèle non disponible: {str(e)}")
    
    # 1. Créer un DataFrame de candidats
    candidates = []
    
    for idx, product in products_df.iterrows():
        candidate = {}
        candidate.update(customer_profile)
        candidate.update(product.to_dict())
        candidates.append(candidate)
    
    candidates_df = pd.DataFrame(candidates)
    
    # 2. Conserver les infos pour l'affichage
    product_ids = candidates_df['product_id'].copy()
    product_prices = candidates_df['price'].copy() if 'price' in candidates_df.columns else None
    
    # 3. Préparer X pour le modèle
    cols_to_drop = ['customer_unique_id', 'product_id', 'label']
    
    X = candidates_df.drop(
        columns=[col for col in cols_to_drop if col in candidates_df.columns],
        errors='ignore'
    )
    
    X = X.fillna(0)
    
    # 4. Prédiction des scores
    try:
        scores = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")
    
    # 5. Construire le résultat
    results = pd.DataFrame({
        'product_id': product_ids,
        'score': scores
    })
    
    if product_prices is not None:
        results['price'] = product_prices.values
    
    # 6. Filtrer par score minimum
    results = results[results["score"] >= min_score]
    
    # 7. Trier et limiter au top-k
    results = results.sort_values("score", ascending=False).head(top_k)
    
    return results.reset_index(drop=True)


# =====================================
# Routes existantes (conservées)
# =====================================

@router.post("/data_proc")
def run_ingestion(req: IngestionRequest):
    """Ingestion et traitement des données (route existante)"""
    summary = run_ingestion_with_summary()
    return {"ok": True, "summary": summary}


@router.post("/predict", response_model=PredictionResponse)
def predict(features: ItemFeatures):
    """Prédiction individuelle pour un item (route existante)"""
    model = get_model()
    
    X = pd.DataFrame([features.dict()])
    score = model.predict_proba(X)[0, 1]
    
    return PredictionResponse(score=float(score))


# =====================================
# Nouvelles routes pour Streamlit
# =====================================

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Vérifie que l'API, le modèle et les données sont opérationnels"""
    model_loaded = False
    data_loaded = False
    
    try:
        model = get_model()
        model_loaded = model is not None
    except Exception:
        pass
    
    try:
        df = load_final_dataset()
        data_loaded = not df.empty
    except Exception:
        pass
    
    status = "healthy" if (model_loaded and data_loaded) else "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        data_loaded=data_loaded,
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/customers", response_model=List[str])
def get_customers():
    """Retourne la liste des IDs clients disponibles depuis le dataset"""
    try:
        df = load_final_dataset()
        customers = get_unique_customers_list(df)
        
        if not customers:
            raise HTTPException(
                status_code=404,
                detail="Aucun client trouvé dans le dataset"
            )
        
        return customers
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/customers/{customer_unique_id}", response_model=CustomerProfile)
def get_customer_profile(customer_unique_id: str):
    """Retourne le profil d'un client spécifique"""
    try:
        df = load_final_dataset()
        profile = get_customer_profile_data(df, customer_unique_id)
        
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"Client {customer_unique_id} non trouvé"
            )
        
        return CustomerProfile(**profile)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/products")
def get_products():
    """Retourne tous les produits disponibles"""
    try:
        df = load_final_dataset()
        products = get_all_products_data(df)
        
        return {
            "total_products": len(products),
            "products": products.to_dict(orient="records")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.post("/recommendations", response_model=RecommendationResponse)
def generate_recommendations(request: RecommendationRequest):
    """
    Génère des recommandations personnalisées pour un client.
    """
    try:
        # Charger le dataset
        df = load_final_dataset()
        
        # Récupérer le profil client
        profile = get_customer_profile_data(df, request.customer_unique_id)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"Client {request.customer_unique_id} non trouvé"
            )
        
        # Récupérer les produits
        products = get_all_products_data(df)
        
        if products.empty:
            raise HTTPException(
                status_code=404,
                detail="Aucun produit disponible"
            )
        
        # Calculer les recommandations
        recommendations_df = compute_recommendations(
            profile,
            products,
            top_k=request.n_recommendations,
            min_score=request.min_score
        )
        
        if recommendations_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Aucune recommandation avec score >= {request.min_score}"
            )
        
        # Formater la réponse
        recommendations_list = []
        for idx, row in recommendations_df.iterrows():
            recommendations_list.append(
                RecommendationItem(
                    rank=len(recommendations_list) + 1,
                    product_id=row["product_id"],
                    price=float(row["price"]),
                    purchase_probability=float(row["score"])
                )
            )
        
        return RecommendationResponse(
            customer_unique_id=request.customer_unique_id,
            total_recommendations=len(recommendations_list),
            generated_at=datetime.utcnow().isoformat(),
            recommendations=recommendations_list
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération des recommandations: {str(e)}"
        )