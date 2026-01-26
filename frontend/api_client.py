"""
frontend/interface/api_client.py

Client for communicating with the FastAPI backend API
"""

import requests
import streamlit as st
from typing import List, Dict, Optional


class APIClient:
    """Client for the recommendation API"""

    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url

    def health_check(self) -> Dict:
        """Checks the health status of the API"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "model_loaded": False,
                "data_loaded": False,
                "error": "Backend not accessible. Ensure that it is started on http://localhost:8000/api/v1",
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "model_loaded": False,
                "data_loaded": False,
                "error": str(e),
            }

    def get_customers(self) -> List[str]:
        """Retrieve the list of customers"""
        try:
            response = requests.get(f"{self.base_url}/customers", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error retrieving clients: {e}")
            return []

    def get_customer_profile(self, customer_unique_id: str) -> Optional[Dict]:
        """Retrieve the profile of a customer"""
        try:
            response = requests.get(
                f"{self.base_url}/customers/{customer_unique_id}", timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error retrieving customer profile: {e}")
            return None

    def generate_recommendations(
        self,
        customer_unique_id: str,
        n_recommendations: int = 10,
        min_score: float = 0.0,
    ) -> Optional[Dict]:
        """Generate recommendations for a customer"""
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
            st.error(f"Error generating recommendations: {e}")
            return None


@st.cache_resource
def get_api_client() -> APIClient:
    """Returns a cached instance of the API client"""
    return APIClient()
