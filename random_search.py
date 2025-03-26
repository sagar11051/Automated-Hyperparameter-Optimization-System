import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def random_search(model, param_distributions, X, y, n_iter=100, cv=5):
    best_score = -np.inf
    best_params = None
    history = []

    skf = StratifiedKFold(n_splits=cv)

    for _ in range(n_iter):
        params = {key: np.random.choice(values) for key, values in param_distributions.items()}
        model.set_params(**params)
        scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)
            for i in range(y_pred.shape[1]):
                scores.append(roc_auc_score(y_test == i, y_pred[:, i]))
        mean_score = np.mean(scores)
        history.append((params, mean_score))
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    return best_params, best_score, history
