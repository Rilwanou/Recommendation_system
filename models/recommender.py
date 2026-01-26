import pandas as pd
import numpy as np


def recommend_items(model, customer_profile, products_df, top_k=5):
    """
    Recommends products for a specific customer.

    Args:
        model: The trained model (classifier with pipeline)
        customer_profile: dict containing customer characteristics
        products_df: DataFrame containing all available products
        top_k: Number of recommendations to return

    Returns:
        DataFrame with the top_k recommended products and their scores
    """

    # 1. Create a candidates DataFrame by combining customer profile + each product
    candidates = []

    for idx, product in products_df.iterrows():
        # Combine the customer profile with the product characteristics
        candidate = {}

        # Add the customer characteristics
        candidate.update(customer_profile)

        # Add the product characteristics
        candidate.update(product.to_dict())

        candidates.append(candidate)

    candidates_df = pd.DataFrame(candidates)

    # 2. Save information for final display
    product_ids = candidates_df["product_id"].copy()
    # product_categories = (
    #    candidates_df["product_category_name_english"].copy()
    #    if "product_category_name_english" in candidates_df.columns
    #    else None
    # )
    product_prices = (
        candidates_df["price"].copy() if "price" in candidates_df.columns else None
    )

    # 3. Prepare X for the model
    # IMPORTANT: Delete ONLY the IDs and the label.
    # The model pipeline handles categorical variable encoding itself
    cols_to_drop = [
        "customer_unique_id",  # customer ID
        "product_id",  # Product ID
        "label",  # Target (if present)
    ]

    X = candidates_df.drop(
        columns=[col for col in cols_to_drop if col in candidates_df.columns],
        errors="ignore",
    )

    # 4. Replace NaNs with appropriate values
    X = X.fillna(0)

    # 5. Prediction of scores
    try:
        scores = model.predict_proba(X)[
            :, 1
        ]  # Probability of positive class (purchase)
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        print(f"Colonnes dans X: {X.columns.tolist()}")
        print(f"Shape de X: {X.shape}")
        print(f"Types de X:\n{X.dtypes}")
        # Fallback: random scores
        scores = np.random.rand(len(X))

    # 6. Create the results DataFrame
    results = pd.DataFrame({"product_id": product_ids, "score": scores})

    # Add only the price (not the category)
    if product_prices is not None:
        results["price"] = product_prices.values

    # 7. Sort by score descending and return the top-k
    results = results.sort_values("score", ascending=False).head(top_k)

    return results.reset_index(drop=True)


def get_customer_statistics(df, customer_unique_id):
    """
    Returns statistics on the customer's purchases.

    Args:
        df: Complete DataFrame
        customer_unique_id: Unique ID of the customer
    Returns:
        dict with statistics
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
