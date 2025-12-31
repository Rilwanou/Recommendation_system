from pydantic import BaseModel


class ItemFeatures(BaseModel):
    price: float
    product_photos_qty: float
    product_weight_g: float
    product_length_cm: float
    product_height_cm: float
    product_width_cm: float
    purchase_count: float
    purchase_count_state: float
    recency_days: float

    customer_state: str
    customer_city: str
    product_category_name_english: str


class PredictionResponse(BaseModel):
    score: float
