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
# -----------------------
def short_id(x: str, left: int = 10, right: int = 6) -> str:
    if not isinstance(x, str):
        return str(x)
    if len(x) <= left + right + 3:
        return x
    return f"{x[:left]}...{x[-right:]}"


def hero():
    st.markdown(
        """
        <div style="padding: 1.2rem; border-radius: 16px; background: #F6F7FB; border: 1px solid #E6E8F0;">
            <h1 style="margin: 0;">🛒 Système de Recommandation Produits</h1>
            <p style="margin-top: 0.35rem; color: #444;">
                Sélectionnez un client pour obtenir des recommandations personnalisées.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def kpi_box(title: str, value: str):
    with st.container(border=True):
        st.caption(title)
        st.markdown(f"### {value}")


# -----------------------
# CACHE (fix UnhashableParamError)
# -----------------------
@st.cache_data(show_spinner=False, ttl=120)
def cached_customers(_api_client):
    return _api_client.get_customers()


@st.cache_data(show_spinner=False, ttl=120)
def cached_profile(_api_client, customer_id: str):
    return _api_client.get_customer_profile(customer_id)


# -----------------------
# Sidebar
# -----------------------
def sidebar_controls(health_status: dict):
    with st.sidebar:
        st.markdown("## ⚙️ Paramètres")
        top_k = st.slider("Top-K recommandations", 1, 20, 10, 1, key="top_k")
        min_score = st.slider("Score minimum", 0.0, 1.0, 0.0, 0.05, key="min_score")

        st.divider()

        with st.expander("✅ Statut système", expanded=True):
            if health_status.get("status") == "healthy":
                st.success("Backend connecté")
                if health_status.get("model_loaded"):
                    st.success("Modèle chargé")
                if health_status.get("data_loaded"):
                    st.success("Données chargées")
            else:
                st.error("Backend indisponible")
                if health_status.get("error"):
                    st.caption(health_status["error"])

            st.caption("Graphiques : " + ("Plotly" if PLOTLY_OK else "Matplotlib"))

        st.divider()
        st.caption("💡 Conseil : commence par **Recommandations**.")

    return top_k, min_score


def backend_gate(health: dict):
    if health.get("status") == "healthy":
        return

    st.error("⚠️ Le backend n'est pas disponible.")
    st.markdown("### 🔧 Démarrer le backend (avec `uv`)")
    st.code(
        "uv run uvicorn backend.src.api.main:app --reload --host 127.0.0.1 --port 8000",
        language="bash",
    )
    if health.get("error"):
        st.markdown("### 📝 Détails")
        st.error(health["error"])

    st.info("Une fois le backend démarré, recharge la page (F5 ou Ctrl+R).")
    st.stop()


# -----------------------
# Sections
# -----------------------
def accueil_section():
    st.markdown("## 👋 Bienvenue")

    st.markdown(
        """
        **Bienvenue dans votre outil de recommandation produits.**  
        Cette application vous aide à proposer **les bons produits, aux bons clients,
        au bon moment**, à partir de leurs comportements d’achat.
        """
    )

    st.write("")

    # Message clé orienté solution
    st.success(
        "🎯 **Problème résolu :** trop de produits, pas assez de pertinence. "
        "Notre moteur classe automatiquement les produits les plus susceptibles d’être achetés."
    )

    st.write("")

    # Directives simples
    st.markdown("### 🚀 Comment utiliser l’outil")
    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown("### ① Recommandations")
            st.write(
                "Sélectionnez un client dans la base et lancez le calcul "
                "pour obtenir un **Top-K personnalisé**."
            )

    with c2:
        with st.container(border=True):
            st.markdown("### ② Résultats")
            st.write(
                "Consultez les produits recommandés, leurs scores et leurs prix. "
                "Les résultats sont **exportables en CSV**."
            )

    with c3:
        with st.container(border=True):
            st.markdown("### ③ Visualisations")
            st.write(
                "Analysez la cohérence des recommandations "
                "(rang, distribution des scores, score vs prix)."
            )

    st.write("")

    # Call to action clair
    st.info(
        "👉 **Commencez par l’onglet _Recommandations_** pour sélectionner un client "
        "et générer vos premières recommandations."
    )


def show_reco_charts(recs: pd.DataFrame):
    if recs is None or recs.empty or "purchase_probability" not in recs.columns:
        st.info(
            "Aucune recommandation à visualiser. Génère d’abord des recommandations."
        )
        return

    st.markdown("## 📊 Visualisations des recommandations")
    st.caption("Analyse rapide de la cohérence des scores sur le Top-K.")

    plot_df = recs.copy()
    if "rank" not in plot_df.columns:
        plot_df["rank"] = range(1, len(plot_df) + 1)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Scores par rang")
        if PLOTLY_OK:
            fig = px.bar(
                plot_df,
                x="rank",
                y="purchase_probability",
                hover_data=[c for c in ["product_id", "price"] if c in plot_df.columns],
                labels={"rank": "Rang", "purchase_probability": "Score"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.bar(plot_df["rank"], plot_df["purchase_probability"])
            ax.set_xlabel("Rang")
            ax.set_ylabel("Score")
            st.pyplot(fig)

        st.caption(
            f"🔎 Score min/max: **{plot_df['purchase_probability'].min():.3f}** → **{plot_df['purchase_probability'].max():.3f}**"
        )

    with col2:
        st.markdown("### Distribution des scores")
        if PLOTLY_OK:
            fig = px.histogram(
                plot_df,
                x="purchase_probability",
                nbins=12,
                labels={"purchase_probability": "Score"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.hist(plot_df["purchase_probability"], bins=12)
            ax.set_xlabel("Score")
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)

        st.caption(
            f"📌 Score médian : **{plot_df['purchase_probability'].median():.3f}**"
        )

    st.write("")
    st.markdown("### Score vs Prix")
    if "price" in plot_df.columns:
        if PLOTLY_OK:
            fig = px.scatter(
                plot_df,
                x="price",
                y="purchase_probability",
                hover_data=["product_id"] if "product_id" in plot_df.columns else None,
                labels={"price": "Prix", "purchase_probability": "Score"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.scatter(plot_df["price"], plot_df["purchase_probability"])
            ax.set_xlabel("Prix")
            ax.set_ylabel("Score")
            st.pyplot(fig)

        cheap = plot_df.sort_values("price").head(3)["purchase_probability"].mean()
        expensive = plot_df.sort_values("price").tail(3)["purchase_probability"].mean()
        st.caption(
            f"💡 Score moyen (3 moins chers) **{cheap:.3f}** vs (3 plus chers) **{expensive:.3f}**"
        )
    else:
        st.info(
            "La colonne `price` n’est pas disponible pour ce jeu de recommandations."
        )


# -----------------------
# Main App
# -----------------------
def run_app():
    st.set_page_config(page_title="Système de Recommandation", layout="wide")
    hero()

    api_client = get_api_client()

    if "recs" not in st.session_state:
        st.session_state["recs"] = None
    if "selected_customer" not in st.session_state:
        st.session_state["selected_customer"] = None

    with st.spinner("Connexion au backend..."):
        health = api_client.health_check()

    top_k, min_score = sidebar_controls(health)
    backend_gate(health)

    tab_home, tab_rec, tab_viz = st.tabs(
        ["🏠 Accueil", "🎯 Recommandations", "📊 Visualisations"]
    )

    with tab_home:
        accueil_section()

    with tab_rec:
        st.markdown("## 🎯 Recommandations personnalisées")

        with st.spinner("Chargement des clients..."):
            customers = cached_customers(api_client)

        if not customers:
            st.error("Aucun client disponible dans la base de données.")
            st.stop()

        st.markdown("### 1️⃣ Sélection du client")
        selected_customer = st.selectbox(
            "Choisissez un client",
            options=customers,
            index=0,
            format_func=lambda x: short_id(x),
            key="customer_select",
        )

        if not selected_customer:
            st.info("Sélectionne un client pour continuer.")
            st.stop()

        st.session_state["selected_customer"] = selected_customer

        st.markdown("### 2️⃣ Profil client")
        with st.spinner("Récupération du profil via l’API..."):
            profile = cached_profile(api_client, selected_customer)

        if profile is None:
            st.error("Impossible de récupérer le profil du client.")
            st.stop()

        p1, p2, p3 = st.columns(3)
        p1.metric("Client", short_id(selected_customer, 12, 6))
        p2.metric("Ville", str(profile.get("customer_city", "N/A")))
        p3.metric("État", str(profile.get("customer_state", "N/A")))

        with st.expander("Voir les détails du profil", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Comportement**")
                st.write(f"- Achats totaux : **{profile.get('purchase_count', 0)}**")
                st.write(f"- Récence : **{profile.get('recency_days', 0)}** jours")
            with c2:
                st.markdown("**Signal modèle**")
                st.write(
                    f"- Score cooc : **{float(profile.get('cooc_score', 0)):.3f}**"
                )

        st.markdown("### 3️⃣ Générer les recommandations")
        run_btn = st.button(
            "🔍 Lancer la recommandation",
            use_container_width=True,
            type="primary",
            key="run_btn",
        )

        if run_btn:
            with st.spinner("Calcul des recommandations..."):
                t0 = time.time()
                result = api_client.generate_recommendations(
                    customer_unique_id=selected_customer,
                    n_recommendations=top_k,
                    min_score=min_score,
                )
                dt = time.time() - t0

            if result is None:
                st.error("Erreur lors de la génération des recommandations.")
                st.stop()

            recs_df = pd.DataFrame(result.get("recommendations", []))
            if not recs_df.empty and "rank" not in recs_df.columns:
                recs_df["rank"] = range(1, len(recs_df) + 1)

            st.session_state["recs"] = recs_df
            st.success(f"✅ Recommandations générées en {dt:.2f}s")

        recs_df = st.session_state.get("recs")

        st.markdown("### 📌 Résultats")
        if recs_df is None:
            st.info(
                "Clique sur **Lancer la recommandation** pour afficher les résultats."
            )
        elif recs_df.empty:
            st.warning("Aucune recommandation ne passe le filtre de score minimum.")
        else:
            best = recs_df.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Top recommandé", short_id(str(best.get("product_id", "")), 10, 6)
            )
            m2.metric(
                "Score (Top 1)", f"{float(best.get('purchase_probability', 0.0)):.4f}"
            )
            m3.metric("Prix (Top 1)", f"{float(best.get('price', 0.0)):.2f}€")

            show_cols = [
                c
                for c in ["rank", "product_id", "price", "purchase_probability"]
                if c in recs_df.columns
            ]
            st.dataframe(recs_df[show_cols], use_container_width=True)

            st.download_button(
                "⬇️ Télécharger (CSV)",
                data=recs_df.to_csv(index=False).encode("utf-8"),
                file_name=f"recommendations_{selected_customer}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_btn",
            )

            st.info("👉 Va dans **Visualisations** pour analyser les scores.")

    with tab_viz:
        show_reco_charts(st.session_state["recs"])


if __name__ == "__main__":
    run_app()
