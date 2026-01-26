"""
backend/src/api/router.py

Routes API pour le système de recommandation Olist
Utilise les fonctions du dossier models/ à la racine
"""

import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


from models.load_data import (
    load_final_dataset,
    get_unique_customers,
    get_customer_profile,
    get_all_products,
)
from models.recommender import recommend_items
from models.load_model import load_trained_model

from services.ingestion import run_ingestion_with_summary
from services.schemas import ItemFeatures, PredictionResponse

# Ajouter le dossier racine au path pour importer depuis models/
root_dir = Path(__file__).resolve().parents[3]


# =====================================
# Router principal
# =====================================

router = APIRouter()


# =====================================
# Schémas Pydantic
# =====================================


class IngestionRequest(BaseModel):
    raw_subdir: str | None = None


class RecommendationRequest(BaseModel):
    customer_unique_id: str = Field(..., description="ID unique du client")
    n_recommendations: int = Field(
        10, ge=1, le=50, description="Nombre de recommandations"
    )
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
# Cache pour modèle et données
# =====================================

_MODEL_CACHE = None
_DATA_CACHE = None


def get_model():
    """Charge et met en cache le modèle"""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = load_trained_model()
    return _MODEL_CACHE


def get_dataset():
    """Charge et met en cache le dataset"""
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = load_final_dataset()
    return _DATA_CACHE


# =====================================
# Routes existantes (conservées)
# =====================================


@router.post("/data_proc")
def run_ingestion(req: IngestionRequest):
    """
    Déclenche le pipeline d’ingestion et de préparation des données
    et retourne un résumé de l’exécution.
    """
    summary = run_ingestion_with_summary()
    return {"ok": True, "summary": summary}


@router.post("/predict", response_model=PredictionResponse)
def predict(features: ItemFeatures):
    """
    Effectue une prédiction de probabilité pour un item
    à partir des caractéristiques fournies.
    """
    model = get_model()

    X = pd.DataFrame([features.dict()])
    score = model.predict_proba(X)[0, 1]

    return PredictionResponse(score=float(score))


# =====================================
# Nouvelles routes pour Streamlit
# =====================================


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Vérifie la disponibilité du modèle entraîné et du jeu de données
    afin d’évaluer l’état global de l’API.
    """
    model_loaded = False
    data_loaded = False

    try:
        model = get_model()
        model_loaded = model is not None
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")

    try:
        df = get_dataset()
        data_loaded = not df.empty
    except Exception as e:
        print(f"❌ Erreur chargement données: {e}")

    status = "healthy" if (model_loaded and data_loaded) else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        data_loaded=data_loaded,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/customers", response_model=List[str])
def get_customers():
    """
    Récupère la liste des identifiants clients uniques présents
    dans le jeu de données chargé.
    """
    try:
        df = get_dataset()
        customers = get_unique_customers(df)

        if not customers:
            raise HTTPException(
                status_code=404, detail="Aucun client trouvé dans le dataset"
            )

        return customers

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/customers/{customer_unique_id}", response_model=CustomerProfile)
def get_customer_profile_route(customer_unique_id: str):
    """
    Retourne le profil détaillé d’un client donné à partir de son identifiant unique.
    """
    try:
        df = get_dataset()
        profile = get_customer_profile(df, customer_unique_id)

        if profile is None:
            raise HTTPException(
                status_code=404, detail=f"Client {customer_unique_id} non trouvé"
            )

        return CustomerProfile(**profile)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/products")
def get_products():
    """
    Retourne la liste de tous les produits disponibles dans le dataset,
    ainsi que le nombre total de produits.
    """
    try:
        df = get_dataset()
        products = get_all_products(df)

        return {
            "total_products": len(products),
            "products": products.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.post("/recommendations", response_model=RecommendationResponse)
def generate_recommendations(request: RecommendationRequest):
    """
    Génère des recommandations de produits pour un client donné
    à partir de son profil, du modèle entraîné et du catalogue produits.
    """
    try:
        # Charger le dataset et le modèle
        df = get_dataset()
        model = get_model()

        # Récupérer le profil client
        profile = get_customer_profile(df, request.customer_unique_id)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"Client {request.customer_unique_id} non trouvé",
            )

        # Récupérer les produits disponibles
        products = get_all_products(df)
        if products.empty:
            raise HTTPException(status_code=404, detail="Aucun produit disponible")

        # Générer les recommandations via le moteur de recommandation
        recommendations_df = recommend_items(
            model, profile, products, top_k=request.n_recommendations
        )

        # Appliquer un seuil minimal sur le score
        if "score" in recommendations_df.columns:
            recommendations_df = recommendations_df[
                recommendations_df["score"] >= request.min_score
            ]

        if recommendations_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Aucune recommandation avec score >= {request.min_score}",
            )

        # Construire la réponse API
        recommendations_list = []
        for _, row in recommendations_df.iterrows():
            recommendations_list.append(
                RecommendationItem(
                    rank=len(recommendations_list) + 1,
                    product_id=row["product_id"],
                    price=float(row["price"]),
                    purchase_probability=float(row["score"]),
                )
            )

        return RecommendationResponse(
            customer_unique_id=request.customer_unique_id,
            total_recommendations=len(recommendations_list),
            generated_at=datetime.utcnow().isoformat(),
            recommendations=recommendations_list,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération des recommandations: {str(e)}",
        )
