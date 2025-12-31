import pandas as pd
from fastapi import APIRouter

from frontend.api.schemas import ItemFeatures, PredictionResponse
from frontend.api.dependencies import get_model

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(features: ItemFeatures):
    model = get_model()

    # Transformer l'entrée en DataFrame (1 ligne)
    X = pd.DataFrame([features.dict()])

    # Probabilité d'achat
    score = model.predict_proba(X)[0, 1]

    return PredictionResponse(score=float(score))
