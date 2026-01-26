import time
import pandas as pd
import streamlit as st

from interface.load_model import load_trained_model
from interface.load_data import (
    load_final_dataset, 
    get_unique_customers, 
    get_customer_profile,
    get_all_products,
    get_customer_history
)
from interface.recommender import recommend_items

# Plotly optionnel (sinon fallback matplotlib)
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

import matplotlib.pyplot as plt


# -----------------------
# Cache & data loading
# -----------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return load_trained_model()


def format_recs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "score" in out.columns:
        out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0).round(4)
    return out


# -----------------------
# UI Helpers
# -----------------------
def hero():
    st.markdown(
        """
        <div style="padding: 1.2rem 1.2rem; border-radius: 16px; background: #F6F7FB; border: 1px solid #E6E8F0;">
            <h1 style="margin: 0;">🛒 Système de Recommandation Produits</h1>
            <p style="margin: 0.4rem 0 0 0; color: #444;">
                Sélectionnez un client pour obtenir des recommandations personnalisées
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def sidebar_controls():
    with st.sidebar:
        st.markdown("## ⚙️ Paramètres")
        top_k = st.slider("Top-K recommandations", 1, 20, 10, 1, key="top_k")
        min_score = st.slider("Score minimum", 0.0, 1.0, 0.0, 0.05, key="min_score")

        st.divider()
        st.markdown("## ✅ Statut")

        # Status modèle
        try:
            _ = get_model()
            st.success("Modèle chargé")
        except Exception:
            st.error("Modèle indisponible")

        # Status Plotly
        if PLOTLY_OK:
            st.success("Graphiques interactifs (Plotly)")
        else:
            st.info("Graphiques (Matplotlib)")

        st.divider()
        st.caption(
            "💡 Sélectionnez un client dans l'onglet **Recommandations** pour voir les produits suggérés."
        )

    return top_k, min_score


def show_reco_charts(recs: pd.DataFrame):
    """Graphiques sur les recommandations."""
    if recs is None or recs.empty or "score" not in recs.columns:
        st.info(
            "Aucune recommandation disponible. Va dans l'onglet **Recommandations** puis sélectionne un client."
        )
        return

    st.markdown("### 📊 Visualisations des recommandations")
    st.caption("Analyse rapide des scores sur le Top-K recommandé.")

    plot_df = recs.reset_index(drop=True).copy()
    plot_df["rank"] = plot_df.index + 1

    col1, col2 = st.columns(2)

    # Bar chart rang vs score
    with col1:
        st.markdown("#### Scores par rang")
        if PLOTLY_OK:
            fig = px.bar(
                plot_df,
                x="rank",
                y="score",
                hover_data=[
                    c
                    for c in ["product_id", "product_category_name_english", "price"]
                    if c in plot_df.columns
                ],
                labels={"rank": "Rang", "score": "Score"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.bar(plot_df["rank"], plot_df["score"])
            ax.set_xlabel("Rang")
            ax.set_ylabel("Score")
            st.pyplot(fig)

    # Histogramme scores
    with col2:
        st.markdown("#### Distribution des scores")
        if PLOTLY_OK:
            fig = px.histogram(plot_df, x="score", nbins=15, labels={"score": "Score"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.hist(plot_df["score"], bins=15)
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
                y="score",
                hover_data=[
                    c
                    for c in ["product_id", "product_category_name_english"]
                    if c in plot_df.columns
                ],
                labels={"price": "Prix", "score": "Score"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.scatter(plot_df["price"], plot_df["score"])
            ax.set_xlabel("Prix")
            ax.set_ylabel("Score")
            st.pyplot(fig)

    # Top catégories
    if "product_category_name_english" in plot_df.columns:
        st.markdown("#### Catégories recommandées (Top-K)")
        counts = plot_df["product_category_name_english"].value_counts().reset_index()
        counts.columns = ["category", "count"]

        if PLOTLY_OK:
            fig = px.bar(
                counts,
                x="category",
                y="count",
                labels={"category": "Catégorie", "count": "Nombre"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.bar(counts["category"], counts["count"])
            ax.set_xlabel("Catégorie")
            ax.set_ylabel("Nombre")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)


def show_data_stats(df: pd.DataFrame):
    st.markdown("### 🔍 Données & statistiques")
    st.caption("Exploration rapide du dataset final.")

    # Infos générales
    c1, c2, c3 = st.columns(3)
    c1.metric("Nb lignes", len(df))
    c2.metric("Nb colonnes", df.shape[1])
    c3.metric("Valeurs manquantes", int(df.isna().sum().sum()))

    with st.expander("👀 Aperçu des données (50 premières lignes)", expanded=True):
        st.dataframe(df.head(50), use_container_width=True)

    with st.expander("📌 Types de variables", expanded=False):
        dtypes_df = pd.DataFrame(
            {"colonne": df.columns, "type": df.dtypes.astype(str)}
        )
        st.dataframe(dtypes_df, use_container_width=True)

    num = df.select_dtypes(include="number")
    with st.expander("📈 Statistiques descriptives (numérique)", expanded=True):
        if num.empty:
            st.warning("Aucune colonne numérique détectée.")
        else:
            st.dataframe(num.describe().T, use_container_width=True)

            col = st.selectbox(
                "Variable à visualiser", list(num.columns), key="num_col_select"
            )
            if PLOTLY_OK:
                fig = px.histogram(
                    df, x=col, nbins=30, title=f"Distribution de {col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots()
                ax.hist(df[col].dropna(), bins=30)
                ax.set_xlabel(col)
                ax.set_ylabel("Fréquence")
                st.pyplot(fig)

            if num.shape[1] >= 2:
                st.markdown("##### Corrélations")
                corr = num.corr(numeric_only=True)
                if PLOTLY_OK:
                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        aspect="auto",
                        title="Matrice de corrélation",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig, ax = plt.subplots()
                    cax = ax.imshow(corr.values)
                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                    ax.set_yticks(range(len(corr.index)))
                    ax.set_yticklabels(corr.index)
                    fig.colorbar(cax)
                    st.pyplot(fig)

    if "product_category_name_english" in df.columns:
        with st.expander("📦 Répartition des catégories", expanded=False):
            counts = (
                df["product_category_name_english"]
                .value_counts()
                .head(15)
                .reset_index()
            )
            counts.columns = ["category", "count"]
            if PLOTLY_OK:
                fig = px.bar(
                    counts,
                    x="category",
                    y="count",
                    title="Top 15 catégories",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots()
                ax.bar(counts["category"], counts["count"])
                ax.set_xlabel("Catégorie")
                ax.set_ylabel("Nombre")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig)


def accueil_section():
    st.markdown("## 👋 Bienvenue")
    st.info(
        """
        **Objectif de cette application :**
        - Sélectionner un client parmi la base de données
        - L'application récupère automatiquement son profil
        - Le modèle génère des recommandations personnalisées
        - Visualiser les résultats

        👉 Commence par l'onglet **Recommandations** pour sélectionner un client.
        """
    )


# -----------------------
# Main App
# -----------------------
def run_app():
    st.set_page_config(page_title="Système de Recommandation", layout="wide")

    hero()

    # Session state
    if "recs" not in st.session_state:
        st.session_state["recs"] = None
    if "selected_customer" not in st.session_state:
        st.session_state["selected_customer"] = None

    # Sidebar
    top_k, min_score = sidebar_controls()

    # Chargement modèle
    try:
        model = get_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        st.stop()

    # Chargement des données
    try:
        df = load_final_dataset()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        st.stop()

    # Liste des clients
    customers = get_unique_customers(df)
    if not customers:
        st.error("Aucun client trouvé dans la base de données (colonne 'customer_unique_id' manquante ou vide).")
        st.stop()

    # Produits disponibles
    products = get_all_products(df)

    # Onglets
    tab_home, tab_rec, tab_viz, tab_data = st.tabs(
        ["🏠 Accueil", "🎯 Recommandations", "📊 Visualisations", "🔍 Données & stats"]
    )

    # Accueil
    with tab_home:
        accueil_section()

    # Recommandations
    with tab_rec:
        st.markdown("## 🎯 Recommandations personnalisées")

        # Sélection du client
        st.markdown("### 1️⃣ Sélection du client")
        selected_customer = st.selectbox(
            "Choisissez un client",
            options=customers,
            index=0,
            key="customer_select"
        )

        if selected_customer:
            st.session_state["selected_customer"] = selected_customer
            
            # Profil client (récupéré automatiquement)
            profile = get_customer_profile(df, selected_customer)
            
            if profile is None:
                st.error("Impossible de récupérer le profil du client.")
                st.stop()
            
            # Affichage simplifié du profil
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

            # Historique (optionnel)
            with st.expander("📜 Historique d'achats", expanded=False):
                history = get_customer_history(df, selected_customer)
                if not history.empty:
                    st.dataframe(history.head(10), use_container_width=True)
                else:
                    st.info("Aucun historique disponible")

            # Recommandations
            st.markdown("### 3️⃣ Générer les recommandations")
            
            st.info(f"📊 Le modèle va analyser **{len(products)}** produits candidats pour ce client.")
            
            run_btn = st.button(
                "🔍 Lancer la recommandation",
                use_container_width=True,
                type="primary",
                key="run_btn",
            )

            if run_btn:
                with st.spinner("Calcul des recommandations en cours..."):
                    t0 = time.time()
                    try:
                        recs = recommend_items(model, profile, products, top_k=top_k)
                        dt = time.time() - t0
                        
                        if "score" in recs.columns:
                            recs = recs[recs["score"] >= min_score].copy()

                        recs = format_recs(recs)
                        st.session_state["recs"] = recs

                        st.success(f"✅ Recommandations générées en {dt:.2f}s")

                        st.markdown("### 📌 Résultats")
                        if recs.empty:
                            st.warning("Aucune recommandation ne passe le filtre de score minimum.")
                        else:
                            best = recs.iloc[0]
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Top recommandé", str(best.get("product_id", "")))
                            c2.metric("Score (Top 1)", f"{float(best.get('score', 0.0)):.4f}")
                            if "price" in recs.columns:
                                c3.metric("Prix (Top 1)", f"{float(best.get('price', 0.0)):.2f}€")

                            st.dataframe(recs, use_container_width=True)

                            st.download_button(
                                "⬇️ Télécharger les recommandations (CSV)",
                                data=recs.to_csv(index=False).encode("utf-8"),
                                file_name=f"recommendations_{selected_customer}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                key="download_btn",
                            )

                            st.info("👉 Va dans l'onglet **Visualisations** pour analyser les scores.")
                    
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération des recommandations: {e}")
                        st.exception(e)

    # Visualisations
    with tab_viz:
        show_reco_charts(st.session_state["recs"])

    # Données & stats
    with tab_data:
        show_data_stats(df)


if __name__ == "__main__":
    run_app()