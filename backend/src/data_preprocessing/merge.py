def build_base_table(dfs: dict):
    """
    Construit la table principale user–item.
    """
    customers = dfs["customers"]
    orders = dfs["orders"]
    order_items = dfs["order_items"]
    products = dfs["products"]
    translation = dfs["category_translation"]

    products = products.merge(translation, on="product_category_name", how="left")

    df = (
        order_items.merge(orders, on="order_id")
        .merge(customers, on="customer_id")
        .merge(products, on="product_id", how="left")
    )

    return df
