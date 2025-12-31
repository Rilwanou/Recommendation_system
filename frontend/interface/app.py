import streamlit as st
from interface.load_model import load_trained_model
from interface.sample_data import get_demo_candidates
from interface.recommender import recommend_items


def run_app():
    st.set_page_config(page_title="Recommendation System Demo", layout="centered")

    st.title("🛒 Recommendation System – Demo")

    st.markdown(
        """
        Cette démo montre comment le modèle recommande
        des produits à partir de leurs caractéristiques.
        """
    )

    # Charger le modèle
    model = load_trained_model()

    # Charger des candidats
    candidates = get_demo_candidates()

    st.subheader("Produits candidats")
    st.dataframe(candidates)

    if st.button("🔍 Recommander"):
        recs = recommend_items(model, candidates, top_k=3)

        st.subheader("📌 Recommandations")
        st.dataframe(recs)
