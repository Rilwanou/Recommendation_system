from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }
