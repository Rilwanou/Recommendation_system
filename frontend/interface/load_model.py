from pathlib import Path
import joblib


def load_trained_model():
    """Charge le modèle entraîné depuis le backend."""
    base_dir = Path(__file__).resolve().parents[2]
    model_path = base_dir / "models" / "logistic_ridge.joblib"

    if not model_path.exists():
        raise FileNotFoundError("Modèle non trouvé. Entraîne le backend d'abord.")

    return joblib.load(model_path)

