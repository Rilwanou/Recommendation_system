from pathlib import Path
import joblib


def load_trained_model():
    """Charge le modèle entraîné depuis le backend."""
    model_path = Path(__file__).resolve().parent / "logistic_ridge.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé : {model_path}")

    return joblib.load(model_path)
