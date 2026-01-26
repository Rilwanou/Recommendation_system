"""
backend/src/services/dependencies.py

Dépendances réutilisables pour l'API FastAPI
Charge et met en cache le modèle ML
"""

import joblib
from pathlib import Path
from functools import lru_cache


# Variable globale pour stocker le modèle une seule fois
_MODEL_CACHE = None


@lru_cache(maxsize=1)
def get_model():
    """
    Charge le modèle ML entraîné.
    Utilise un cache pour éviter de recharger le modèle à chaque requête.
    
    Returns:
        Le modèle ML chargé (sklearn LogisticRegression ou autre)
        
    Raises:
        FileNotFoundError: Si le modèle n'existe pas
        Exception: Si le chargement échoue
    """
    global _MODEL_CACHE
    
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    
    # Trouver le chemin du modèle
    # Adapter selon ton architecture de fichiers
    base_dir = Path(__file__).resolve().parents[2]  # Remonte de 2 niveaux
    model_path = base_dir / "models" / "logistic_ridge.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable à {model_path}. "
            f"Assurez-vous d'avoir entraîné le modèle d'abord."
        )
    
    try:
        print(f"📥 Chargement du modèle depuis: {model_path}")
        _MODEL_CACHE = joblib.load(model_path)
        print("✅ Modèle chargé avec succès")
        return _MODEL_CACHE
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        raise


def reload_model():
    """
    Force le rechargement du modèle.
    Utile après un réentraînement.
    """
    global _MODEL_CACHE
    _MODEL_CACHE = None
    get_model.cache_clear()
    return get_model()
