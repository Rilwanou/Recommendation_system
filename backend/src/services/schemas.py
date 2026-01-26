"""
backend/src/services/schemas.py

Schémas Pydantic pour la validation des données
"""

from pydantic import BaseModel, Field
from typing import Optional


class ItemFeatures(BaseModel):
    """
    Features d'un item pour la prédiction individuelle.
    Utilisé par la route /predict
    """
    price: float = Field(..., description="Prix du produit")
    product_photos_qty: int = Field(..., description="Nombre de photos")
    product_weight_g: float = Field(..., description="Poids en grammes")
    product_length_cm: float = Field(..., description="Longueur en cm")
    product_height_cm: float = Field(..., description="Hauteur en cm")
    product_width_cm: float = Field(..., description="Largeur en cm")
    purchase_count: int = Field(..., description="Nombre d'achats total")
    purchase_count_state: int = Field(..., description="Nombre d'achats dans l'état")
    recency_days: int = Field(..., description="Jours depuis dernier achat")
    
    # Optionnels pour enrichissement
    customer_state: Optional[str] = Field(None, description="État du client")
    customer_city: Optional[str] = Field(None, description="Ville du client")
    product_category_name_english: Optional[str] = Field(None, description="Catégorie produit")

    class Config:
        schema_extra = {
            "example": {
                "price": 59.9,
                "product_photos_qty": 3,
                "product_weight_g": 500.0,
                "product_length_cm": 20.0,
                "product_height_cm": 10.0,
                "product_width_cm": 15.0,
                "purchase_count": 120,
                "purchase_count_state": 40,
                "recency_days": 30,
                "customer_state": "SP",
                "customer_city": "sao paulo",
                "product_category_name_english": "housewares"
            }
        }


class PredictionResponse(BaseModel):
    """
    Réponse de prédiction pour un item individuel.
    Utilisé par la route /predict
    """
    score: float = Field(..., description="Probabilité d'achat [0-1]", ge=0.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "score": 0.753
            }
        }
