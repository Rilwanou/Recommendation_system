def recommend_items(model, candidates_df, top_k=5):
    """
    Calcule un score de recommandation et retourne le Top-K.
    """
    X = candidates_df.drop(columns=["item_id"])

    scores = model.predict_proba(X)[:, 1]

    results = candidates_df.copy()
    results["score"] = scores

    return results.sort_values("score", ascending=False).head(top_k)[
        ["item_id", "score", "product_category_name_english", "price"]
    ]
