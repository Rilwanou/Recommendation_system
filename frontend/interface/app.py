import time
import pandas as pd
import streamlit as st

from interface.load_model import load_trained_model
from interface.sample_data import get_demo_candidates
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


@st.cache_data(show_spinner=False)
def get_candidates():
    return get_demo_candidates()


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
            <h1 style="margin: 0;">🛒 Recommendation System — Demo</h1>
            <p style="margin: 0.4rem 0 0 0; color: #444;">
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def sidebar_controls():
    with st.sidebar:
        st.markdown("## ⚙️ Paramètres")
        top_k = st.slider("Top-K recommandations", 1, 20, 5, 1, key="top_k")
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
            "💡 Astuce: calcule les recommandations dans l’onglet **Recommandations** pour alimenter **Visualisations**."
        )

    return top_k, min_score


def show_reco_charts(recs: pd.DataFrame):
    """Graphiques sur les recommandations."""
    if recs is None or recs.empty or "score" not in recs.columns:
        st.info(
            "Aucune recommandation disponible. Va dans l’onglet **Recommandations** puis clique sur **Recommander**."
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
                    for c in ["item_id", "product_category_name_english", "price"]
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
                    for c in ["item_id", "product_category_name_english"]
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


def show_data_stats(candidates: pd.DataFrame):
    st.markdown("### 🔍 Données & statistiques")
    st.caption("Exploration rapide des données candidates utilisées par la démo.")

    # Infos générales
    c1, c2, c3 = st.columns(3)
    c1.metric("Nb lignes", len(candidates))
    c2.metric("Nb colonnes", candidates.shape[1])
    c3.metric("Valeurs manquantes", int(candidates.isna().sum().sum()))

    with st.expander("👀 Aperçu des données (50 premières lignes)", expanded=True):
        st.dataframe(candidates.head(50), use_container_width=True)

    with st.expander("📌 Types de variables", expanded=False):
        dtypes_df = pd.DataFrame(
            {"colonne": candidates.columns, "type": candidates.dtypes.astype(str)}
        )
        st.dataframe(dtypes_df, use_container_width=True)

    num = candidates.select_dtypes(include="number")
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
                    candidates, x=col, nbins=30, title=f"Distribution de {col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots()
                ax.hist(candidates[col].dropna(), bins=30)
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

    if "product_category_name_english" in candidates.columns:
        with st.expander("📦 Répartition des catégories", expanded=False):
            counts = (
                candidates["product_category_name_english"]
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
                    title="Top 15 catégories (candidats)",
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
        - Afficher des produits candidats
        - Calculer un score de recommandation (via le modèle)
        - Retourner un classement Top-K
        - Visualiser rapidement les résultats

        👉 Commence par l’onglet **Recommandations**, puis explore **Visualisations**.
        """
    )


# -----------------------
# Main App
# -----------------------
def run_app():
    st.set_page_config(page_title="Recommendation System Demo", layout="wide")

    hero()

    # Session state pour partager les recos entre onglets
    if "recs" not in st.session_state:
        st.session_state["recs"] = None

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

    # Données (démo)
    candidates = get_candidates()
    if "item_id" not in candidates.columns:
        st.error("La colonne `item_id` est obligatoire dans les candidats.")
        st.stop()

    # Onglets
    tab_home, tab_rec, tab_viz, tab_data = st.tabs(
        ["🏠 Accueil", "🎯 Recommandations", "📊 Visualisations", "🔍 Données & stats"]
    )

    # Accueil
    with tab_home:
        accueil_section()

    # Recommandations
    with tab_rec:
        st.markdown("## 🎯 Recommandations")

        st.caption("Filtre optionnel par catégorie, puis clique sur **Recommander**.")

        candidates_view = candidates.copy()
        if "product_category_name_english" in candidates.columns:
            categories = ["(toutes)"] + sorted(
                candidates["product_category_name_english"].dropna().unique().tolist()
            )
            chosen_cat = st.selectbox(
                "Filtrer par catégorie", categories, index=0, key="cat_filter"
            )
            if chosen_cat != "(toutes)":
                candidates_view = candidates[
                    candidates["product_category_name_english"] == chosen_cat
                ].copy()

        col_left, col_right = st.columns([1.25, 1])
        with col_left:
            st.subheader("Produits candidats")
            st.dataframe(candidates_view, use_container_width=True)
        with col_right:
            st.subheader("Actions")
            st.write("Paramètres actuels:")
            st.code(f"Top-K = {top_k}\nScore min = {min_score}", language="text")
            run_btn = st.button(
                "🔍 Recommander",
                use_container_width=True,
                type="primary",
                key="run_btn",
            )

        if run_btn:
            with st.spinner("Calcul des recommandations..."):
                t0 = time.time()
                recs = recommend_items(model, candidates_view, top_k=top_k)
                dt = time.time() - t0

            if "score" in recs.columns:
                recs = recs[recs["score"] >= min_score].copy()

            recs = format_recs(recs)
            st.session_state["recs"] = recs

            st.success(f"Recommandations générées en {dt:.2f}s ✅")

            st.subheader("📌 Résultats")
            if recs.empty:
                st.warning("Aucune recommandation ne passe le filtre de score minimum.")
            else:
                best = recs.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Top recommandé", str(best.get("item_id", "")))
                c2.metric("Score (Top 1)", float(best.get("score", 0.0)))
                if "price" in recs.columns:
                    c3.metric("Prix (Top 1)", float(best.get("price", 0.0)))

                st.dataframe(recs, use_container_width=True)

                st.download_button(
                    "⬇️ Télécharger les recommandations (CSV)",
                    data=recs.to_csv(index=False).encode("utf-8"),
                    file_name="recommendations.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_btn",
                )

                st.info(
                    "👉 Va dans l’onglet **Visualisations** pour analyser les scores."
                )

    # Visualisations
    with tab_viz:
        show_reco_charts(st.session_state["recs"])

    # Données & stats
    with tab_data:
        show_data_stats(candidates)


if __name__ == "__main__":
    run_app()
