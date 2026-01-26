import time
import pandas as pd
import streamlit as st
from api_client import get_api_client

# Plotly optionnel
try:
    import plotly.express as px

    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

import matplotlib.pyplot as plt


# -----------------------
# UI Helpers
def hero(show_client_hint=False):
    st.markdown(
        f"""
        <div style="padding: 1.2rem 1.2rem; border-radius: 16px; background: #F6F7FB; border: 1px solid #E6E8F0;">
            <h1 style="margin: 0; color: #000;">🛒 Système de Recommandation Produits</h1>
            {"<p style='margin: 0.4rem 0 0 0; color: #000;'>Sélectionnez un client pour obtenir des recommandations personnalisées</p>" if show_client_hint else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def sidebar_controls(health_status):
    with st.sidebar:
        st.markdown("## ✅ Statut Backend")

        # Health check
        if health_status["status"] == "healthy":
            st.success("✅ API opérationnelle")
            if health_status.get("model_loaded"):
                st.success("✅ Modèle chargé")
            if health_status.get("data_loaded"):
                st.success("✅ Données chargées")
        else:
            st.error("❌ Backend non disponible")
            if "error" in health_status:
                st.caption(f"Erreur: {health_status['error']}")

        st.divider()
        st.caption("💡 Sélectionnez un client dans l'onglet **Recommandations**")


def show_reco_charts(recs: pd.DataFrame):
    """Graphiques sur les recommandations."""
    if recs is None or recs.empty or "purchase_probability" not in recs.columns:
        st.info(
            "Aucune recommandation disponible. Va dans l'onglet **Recommandations** puis sélectionne un client."
        )
        return

    st.markdown("### 📊 Visualisations des recommandations")
    st.caption("Analyse rapide des scores sur le Top-K recommandé.")

    plot_df = recs.copy()

    col1, col2 = st.columns(2)

    # Bar chart rang vs score
    with col1:
        st.markdown("#### Scores par rang")
        if PLOTLY_OK:
            fig = px.bar(
                plot_df,
                x="rank",
                y="purchase_probability",
                hover_data=["product_id", "price"],
                labels={"rank": "Rang", "purchase_probability": "Score"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.bar(plot_df["rank"], plot_df["purchase_probability"])
            ax.set_xlabel("Rang")
            ax.set_ylabel("Score")
            st.pyplot(fig)

    # Histogramme scores
    with col2:
        st.markdown("#### Distribution des scores")
        if PLOTLY_OK:
            fig = px.histogram(
                plot_df,
                x="purchase_probability",
                nbins=15,
                labels={"purchase_probability": "Score"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.hist(plot_df["purchase_probability"], bins=15)
            ax.set_xlabel("Score")
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)

    # Scatter score vs prix
    if "price" in plot_df.columns:
        st.markdown("#### Score vs Prix")
        if PLOTLY_OK:
            fig = px.scatter(
                plot_df,
                x="price",
                y="purchase_probability",
                hover_data=["product_id"],
                labels={"price": "Prix", "purchase_probability": "Score"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.scatter(plot_df["price"], plot_df["purchase_probability"])
            ax.set_xlabel("Prix")
            ax.set_ylabel("Score")
            st.pyplot(fig)


def accueil_section():
    st.markdown("## 👋 Bienvenue")
    st.info(
        """
        **Objectif de cette application :**
        - Sélectionner un client parmi la base de données
        - L'application récupère automatiquement son profil via l'API
        - Le modèle génère des recommandations personnalisées
        - Visualiser les résultats

        👉 Commence par l'onglet **Recommandations** pour sélectionner un client.
        
        **Architecture :**
        - 🔙 Backend: FastAPI (port 8000) - Modèle ML + Données
        - 🎨 Frontend: Streamlit (port 8501) - Interface uniquement
        """
    )


# -----------------------
# Main App
# -----------------------
def run_app():
    st.set_page_config(page_title="Système de Recommandation", layout="wide")

    hero(show_client_hint=False)

    # Initialiser le client API
    api_client = get_api_client()

    # Session state
    if "recs" not in st.session_state:
        st.session_state["recs"] = None
    if "selected_customer" not in st.session_state:
        st.session_state["selected_customer"] = None

    # ⚠️ VÉRIFICATION CRITIQUE : Backend doit être lancé
    with st.spinner("Connexion au backend..."):
        health = api_client.health_check()

    # ❌ Bloquer l'app si backend non disponible
    if health["status"] != "healthy":
        st.error(
            "⚠️ **Le backend n'est pas disponible ou n'a pas démarré correctement**"
        )

        st.markdown("### 🔧 Pour démarrer le backend :")
        st.code(
            """
cd backend
uvicorn backend.src.api.main:app --reload
        """,
            language="bash",
        )

        st.markdown("### 📝 Détails :")
        if "error" in health:
            st.error(health["error"])

        st.info("Une fois le backend démarré, rechargez cette page (F5 ou Ctrl+R)")
        st.stop()  # ⛔ Arrêter complètement l'exécution

    # ✅ Backend OK, on continue
    st.success("✅ Backend connecté et opérationnel")

    # Onglets
    tab_home, tab_rec, tab_viz = st.tabs(
        ["🏠 Accueil", "🎯 Recommandations", "📊 Visualisations"]
    )

    # Accueil
    with tab_home:
        accueil_section()

    # Recommandations
    with tab_rec:
        st.markdown("## 🎯 Recommandations personnalisées")
        with st.expander("⚙️ Paramètres de recommandation", expanded=True):
            top_k = st.slider("Top-K recommandations", 1, 20, 10, 1, key="top_k")
            min_score = st.slider("Score minimum", 0.0, 1.0, 0.0, 0.05, key="min_score")

        # Récupérer la liste des clients
        with st.spinner("Chargement des clients..."):
            customers = api_client.get_customers()

        if not customers:
            st.error("Aucun client disponible dans la base de données.")
            st.stop()

        # Sélection du client
        st.markdown("### 1️⃣ Sélection du client")
        selected_customer = st.selectbox(
            "Choisissez un client", options=customers, index=0, key="customer_select"
        )

        if selected_customer:
            st.session_state["selected_customer"] = selected_customer

            # Récupérer le profil via API
            with st.spinner("Récupération du profil..."):
                profile = api_client.get_customer_profile(selected_customer)

            if profile is None:
                st.error("Impossible de récupérer le profil du client.")
                st.stop()

            # Affichage du profil
            st.markdown("### 2️⃣ Profil client")
            st.success(f"✅ Client **{selected_customer}** sélectionné")

            with st.expander("Voir les détails du profil", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Informations géographiques**")
                    st.text(f"État: {profile.get('customer_state', 'N/A')}")
                    st.text(f"Ville: {profile.get('customer_city', 'N/A')}")

                with col2:
                    st.markdown("**Informations comportementales**")
                    st.text(f"Achats totaux: {profile.get('purchase_count', 0)}")
                    st.text(f"Score cooc: {profile.get('cooc_score', 0):.3f}")
                    st.text(f"Dernier achat: {profile.get('recency_days', 0)} jours")

            # Recommandations
            st.markdown("### 3️⃣ Générer les recommandations")

            run_btn = st.button(
                "🔍 Lancer la recommandation",
                use_container_width=True,
                type="primary",
                key="run_btn",
            )

            if run_btn:
                with st.spinner("Calcul des recommandations en cours..."):
                    t0 = time.time()

                    # Appel API
                    result = api_client.generate_recommendations(
                        customer_unique_id=selected_customer,
                        n_recommendations=top_k,
                        min_score=min_score,
                    )

                    dt = time.time() - t0

                if result is None:
                    st.error("❌ Erreur lors de la génération des recommandations")
                    st.stop()

                # Convertir en DataFrame
                recs_df = pd.DataFrame(result["recommendations"])
                st.session_state["recs"] = recs_df

                st.success(f"✅ Recommandations générées en {dt:.2f}s")

                st.markdown("### 📌 Résultats")
                if recs_df.empty:
                    st.warning(
                        "Aucune recommandation ne passe le filtre de score minimum."
                    )
                else:
                    best = recs_df.iloc[0]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Top recommandé", str(best["product_id"]))
                    c2.metric("Score (Top 1)", f"{best['purchase_probability']:.4f}")
                    c3.metric("Prix (Top 1)", f"{best['price']:.2f}€")

                    st.dataframe(recs_df, use_container_width=True)

                    st.download_button(
                        "⬇️ Télécharger les recommandations (CSV)",
                        data=recs_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"recommendations_{selected_customer}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_btn",
                    )

                    st.info(
                        "👉 Va dans l'onglet **Visualisations** pour analyser les scores."
                    )

    # Visualisations
    with tab_viz:
        show_reco_charts(st.session_state["recs"])


if __name__ == "__main__":
    run_app()
