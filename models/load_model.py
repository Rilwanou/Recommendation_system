from pathlib import Path
import joblib


def load_trained_model():
    """Loads the trained model from the backend."""
    model_path = Path(__file__).resolve().parent / "logistic_ridge.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)
