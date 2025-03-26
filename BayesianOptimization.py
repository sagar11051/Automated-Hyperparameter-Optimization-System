from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class BayesianOptimization:
    def __init__(self, model, param_distributions, X, y, n_iter=25, cv=5):
        self.model = model
        self.param_distributions = param_distributions
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.cv = cv
        self.gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
        self.history = []

    def propose_location(self):
        best_score = -np.inf
        best_params = None
        for _ in range(100):
            params = {key: np.random.choice(values) for key, values in self.param_distributions.items()}
            if 'classifier__max_depth' in params and params['classifier__max_depth'] is None:
                params['classifier__max_depth'] = 1000
            
            if len(self.history) > 0:
                mean, std = self.gpr.predict([list(params.values())], return_std=True)
                score = mean + 1.96 * std
                if score > best_score:
                    best_score = score
                    best_params = params
            else:
                best_params = params

        return best_params

    def optimize(self):
        for _ in range(self.n_iter):
            if len(self.history) > 0:
                X_train = np.array([list(params.values()) for params, score in self.history])
                y_train = np.array([score for params, score in self.history])
                self.gpr.fit(X_train, y_train)
            
            params = self.propose_location()
            self.model.set_params(**params)
            scores = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring='accuracy')
            mean_score = np.mean(scores)
            self.history.append((params, mean_score))
        
        best_params, best_score = max(self.history, key=lambda item: item[1])
        return best_params, best_score
