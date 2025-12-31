from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_model(preprocessor, X, y):
    """
    Entraîne un modèle de régression logistique régularisée (Ridge)
    avec sélection du paramètre C par validation croisée.
    """

    # Pipeline complet
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    # Grille de recherche (issue du notebook)
    param_grid = {"clf__C": [0.03, 0.05, 0.07]}

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",  # métrique RS
        cv=5,
        n_jobs=-1,
    )

    grid_search.fit(X, y)

    print("✅ Best C:", grid_search.best_params_["clf__C"])
    print("✅ Best CV Average Precision:", round(grid_search.best_score_, 4))

    # On retourne le meilleur pipeline entraîné
    return grid_search.best_estimator_
