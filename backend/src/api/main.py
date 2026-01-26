"""
backend/src/api/main.py

Point d'entrée principal de l'API FastAPI
Système de recommandation Olist - Master 2 SEP
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router


# =====================================
# Configuration de l'application
# =====================================

app = FastAPI(
    title="Olist Recommendation System API",
    version="1.0.0",
    description="API REST pour le système de recommandation"
)


# =====================================
# Middleware CORS
# =====================================

# Permet les requêtes depuis Streamlit et autres frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit local
        "http://localhost:3000",  # React local (si besoin)
        "*",  # En production, restreindre aux domaines autorisés
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# Inclusion des routes
# =====================================

app.include_router(
    router,
    prefix="/api/v1",
    tags=["API v1"]
)


# =====================================
# Route racine
# =====================================

@app.get("/", tags=["Root"])
def read_root():
    """
    Route racine - Informations sur l'API
    """
    return {
        "message": "Olist Recommendation System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "data_ingestion": "/api/v1/data_proc",
            "single_prediction": "/api/v1/predict",
            "recommendations": "/api/v1/recommendations",
            "model_info": "/api/v1/model/info",
            "customers": "/api/v1/customers",
            "categories": "/api/v1/categories",
            "candidates": "/api/v1/candidates",
        }
    }