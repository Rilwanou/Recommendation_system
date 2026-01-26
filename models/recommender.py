import pandas as pd
import numpy as np


def recommend_items(model, customer_profile, products_df, top_k=5):
    """
    Recommande des produits pour un client donné.

    Args:
        model: Le modèle entraîné (classifier avec pipeline)
        customer_profile: dict contenant les caractéristiques du client
        products_df: DataFrame contenant tous les produits disponibles
        top_k: Nombre de recommandations à retourner

    Returns:
        DataFrame avec les top_k produits recommandés et leurs scores
    """

    # 1. Créer un DataFrame de candidats en combinant profil client + chaque produit
    candidates = []

    for idx, product in products_df.iterrows():
        # Combiner le profil client avec les caractéristiques du produit
        candidate = {}

        # Ajouter les caractéristiques du client
        candidate.update(customer_profile)

        # Ajouter les caractéristiques du produit
        candidate.update(product.to_dict())

        candidates.append(candidate)

    candidates_df = pd.DataFrame(candidates)

    # 2. Conserver les infos pour l'affichage final
    product_ids = candidates_df["product_id"].copy()
    product_categories = (
        candidates_df["product_category_name_english"].copy()
        if "product_category_name_english" in candidates_df.columns
        else None
    )
    product_prices = (
        candidates_df["price"].copy() if "price" in candidates_df.columns else None
    )

    # 3. Préparer X pour le modèle
    # IMPORTANT : Ne supprimer QUE les IDs et le label
    # Le pipeline du modèle gère lui-même l'encodage des variables catégorielles
    cols_to_drop = [
        "customer_unique_id",  # ID client
        "product_id",  # ID produit
        "label",  # Target (si présente)
    ]

    X = candidates_df.drop(
        columns=[col for col in cols_to_drop if col in candidates_df.columns],
        errors="ignore",
    )

    # 4. Remplacer les NaN par des valeurs appropriées
    X = X.fillna(0)

    # 5. Prédiction des scores
    try:
        scores = model.predict_proba(X)[:, 1]  # Probabilité classe positive (achat)
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        print(f"Colonnes dans X: {X.columns.tolist()}")
        print(f"Shape de X: {X.shape}")
        print(f"Types de X:\n{X.dtypes}")
        # Fallback : scores aléatoires
        scores = np.random.rand(len(X))

    # 6. Créer le DataFrame de résultats
    results = pd.DataFrame({"product_id": product_ids, "score": scores})

    # Ajouter seulement le prix (pas la catégorie)
    if product_prices is not None:
        results["price"] = product_prices.values

    # 7. Trier par score décroissant et retourner le top-k
    results = results.sort_values("score", ascending=False).head(top_k)

    return results.reset_index(drop=True)


def get_customer_statistics(df, customer_unique_id):
    """
    Retourne des statistiques sur les achats du client.

    Args:
        df: DataFrame complet
        customer_unique_id: ID unique du client

    Returns:
        dict avec statistiques
    """
    customer_data = df[df["customer_unique_id"] == customer_unique_id]

    stats = {
        "total_purchases": len(customer_data),
        "total_spent": customer_data["price"].sum()
        if "price" in customer_data.columns
        else 0,
        "avg_price": customer_data["price"].mean()
        if "price" in customer_data.columns
        else 0,
        "favorite_category": customer_data["product_category_name_english"].mode()[0]
        if "product_category_name_english" in customer_data.columns
        and len(customer_data) > 0
        else "N/A",
        "recent_purchases": customer_data["recency_days"].min()
        if "recency_days" in customer_data.columns
        else None,
        "conversion_rate": customer_data["label"].mean()
        if "label" in customer_data.columns
        else None,
    }

    return stats
