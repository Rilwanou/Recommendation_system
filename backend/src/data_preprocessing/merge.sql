 -- 1) Charger les CSV clean
CREATE OR REPLACE TABLE orders AS
SELECT * FROM read_csv_auto('data/processed/orders_clean.csv', header=true);

CREATE OR REPLACE TABLE customers AS
SELECT * FROM read_csv_auto('data/processed/customers_clean.csv', header=true);

CREATE OR REPLACE TABLE order_items AS
SELECT * FROM read_csv_auto('data/processed/order_items_clean.csv', header=true);

CREATE OR REPLACE TABLE products AS
SELECT * FROM read_csv_auto('data/processed/products_clean.csv', header=true);

CREATE OR REPLACE TABLE sellers AS
SELECT * FROM read_csv_auto('data/processed/sellers_clean.csv', header=true);

CREATE OR REPLACE TABLE order_payments AS
SELECT * FROM read_csv_auto('data/processed/order_payments_clean.csv', header=true);

CREATE OR REPLACE TABLE order_reviews AS
SELECT * FROM read_csv_auto('data/processed/order_reviews_clean.csv', header=true);

CREATE OR REPLACE TABLE geolocation AS
SELECT * FROM read_csv_auto('data/processed/geolocation_clean.csv', header=true);

-- 2) Agrégations pour éviter la duplication (1 ligne par order_id)
CREATE OR REPLACE TABLE order_items_agg AS
SELECT
  oi.order_id,
  COUNT(*) AS nb_items,
  COUNT(DISTINCT oi.product_id) AS nb_distinct_products,
  COUNT(DISTINCT oi.seller_id) AS nb_distinct_sellers,
  SUM(oi.price) AS items_price_sum,
  SUM(oi.freight_value) AS freight_sum
FROM order_items oi
GROUP BY oi.order_id;

CREATE OR REPLACE TABLE payments_agg AS
SELECT
  order_id,
  SUM(payment_value) AS payment_value_sum,
  COUNT(*) AS nb_payments,
  MAX(payment_installments) AS max_installments
FROM order_payments
GROUP BY order_id;

CREATE OR REPLACE TABLE reviews_agg AS
SELECT
  order_id,
  AVG(review_score) AS review_score_avg,
  COUNT(*) AS nb_reviews
FROM order_reviews
GROUP BY order_id;

CREATE OR REPLACE TABLE products_features_by_order AS
SELECT
  oi.order_id,
  AVG(p.product_weight_g) AS avg_product_weight_g,
  AVG(p.product_length_cm * p.product_height_cm * p.product_width_cm) AS avg_product_volume_cm3,
  COUNT(DISTINCT p.product_category_name) AS nb_categories
FROM order_items oi
LEFT JOIN products p ON p.product_id = oi.product_id
GROUP BY oi.order_id;

-- 3) Table finale
CREATE OR REPLACE TABLE final_table AS
SELECT
  o.*,
  c.customer_unique_id,
  c.customer_zip_code_prefix,
  c.customer_city,
  c.customer_state,

  ia.nb_items,
  ia.nb_distinct_products,
  ia.nb_distinct_sellers,
  ia.items_price_sum,
  ia.freight_sum,

  pa.payment_value_sum,
  pa.nb_payments,
  pa.max_installments,

  ra.review_score_avg,
  ra.nb_reviews,

  pf.avg_product_weight_g,
  pf.avg_product_volume_cm3,
  pf.nb_categories
FROM orders o
LEFT JOIN customers c ON c.customer_id = o.customer_id
LEFT JOIN order_items_agg ia ON ia.order_id = o.order_id
LEFT JOIN payments_agg pa ON pa.order_id = o.order_id
LEFT JOIN reviews_agg ra ON ra.order_id = o.order_id
LEFT JOIN products_features_by_order pf ON pf.order_id = o.order_id;

-- 4) Export
COPY final_table TO 'data/processed/master_orders.csv' (FORMAT CSV, HEADER);
