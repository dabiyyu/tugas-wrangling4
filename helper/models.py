from sklearn import linear_model

LINEAR_MODEL_CLF = {
    "logreg_cv": linear_model.LogisticRegressionCV(
        max_iter=1000,
        cv=2,
        class_weight='balanced',
        scoring='roc_auc',
        random_state=123,
        solver="liblinear",
        n_jobs=-1
    )
}