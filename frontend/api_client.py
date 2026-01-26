"""
frontend/interface/api_client.py

Client pour communiquer avec l'API FastAPI backend
"""

import requests
import streamlit as st
from typing import List, Dict, Optional


class APIClient:
    """Client pour l'API de recommandation"""

    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url

    def health_check(self) -> Dict:
        """Vérifie l'état de santé de l'API"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "model_loaded": False,
                "data_loaded": False,
                "error": "Backend non accessible. Assurez-vous qu'il est démarré sur http://localhost:8000/api/v1",
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "model_loaded": False,
                "data_loaded": False,
                "error": str(e),
            }

    def get_customers(self) -> List[str]:
        """Récupère la liste des clients"""
        try:
            response = requests.get(f"{self.base_url}/customers", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la récupération des clients: {e}")
            return []

    def get_customer_profile(self, customer_unique_id: str) -> Optional[Dict]:
        """Récupère le profil d'un client"""
        try:
            response = requests.get(
                f"{self.base_url}/customers/{customer_unique_id}", timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la récupération du profil: {e}")
            return None

    def generate_recommendations(
        self,
        customer_unique_id: str,
        n_recommendations: int = 10,
        min_score: float = 0.0,
    ) -> Optional[Dict]:
        """Génère des recommandations pour un client"""
        try:
            payload = {
                "customer_unique_id": customer_unique_id,
                "n_recommendations": n_recommendations,
                "min_score": min_score,
            }

            response = requests.post(
                f"{self.base_url}/recommendations", json=payload, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la génération des recommandations: {e}")
            return None


@st.cache_resource
def get_api_client() -> APIClient:
    """Retourne une instance du client API (cached)"""
    return APIClient()
