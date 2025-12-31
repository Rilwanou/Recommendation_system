import pandas as pd


def get_demo_candidates():
    """
    Génère un petit jeu de données fictif
    pour illustrer les recommandations.
    """
    data = [
        {
            "price": 59.9,
            "product_photos_qty": 3,
            "product_weight_g": 500,
            "product_length_cm": 20,
            "product_height_cm": 10,
            "product_width_cm": 15,
            "purchase_count": 120,
            "purchase_count_state": 40,
            "recency_days": 30,
            "customer_state": "SP",
            "customer_city": "sao paulo",
            "product_category_name_english": "housewares",
            "item_id": "item_1",
        },
        {
            "price": 199.9,
            "product_photos_qty": 1,
            "product_weight_g": 2500,
            "product_length_cm": 40,
            "product_height_cm": 20,
            "product_width_cm": 30,
            "purchase_count": 20,
            "purchase_count_state": 5,
            "recency_days": 120,
            "customer_state": "SP",
            "customer_city": "sao paulo",
            "product_category_name_english": "electronics",
            "item_id": "item_2",
        },
    ]

    return pd.DataFrame(data)
