from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def run_optimization(model, param_distributions, X, y, method='bayesian', n_iter=25, cv=5):
    if method == 'random':
        return random_search(model, param_distributions, X, y, n_iter=n_iter, cv=cv)
    elif method == 'bayesian':
        optimizer = BayesianOptimization(model, param_distributions, X, y, n_iter=n_iter, cv=cv)
        return optimizer.optimize()
    else:
        raise ValueError("Unsupported optimization method")
