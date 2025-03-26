from preprocess import  preprocess_data
from integrate import run_optimization
from BayesianOptimization import *
from random_search import *
from plot import *
from sklearn.datasets import load_iris, load_breast_cancer

def main():
    datasets = {
        'iris': load_iris,
        'breast_cancer': load_breast_cancer
    }

    print("Available datasets:")
    for key in datasets.keys():
        print(key)
    
    dataset_choice = input("Choose a dataset: ")
    if dataset_choice not in datasets:
        print("Invalid choice")
        return
    
    data = datasets[dataset_choice]()
    X, y = data.data, data.target
    X_preprocessed = preprocess_data(X)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])

    param_distributions = {
        'classifier__n_estimators': [10, 50, 100],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }

    print("Optimization methods: random, bayesian")
    method_choice = input("Choose an optimization method: ")

    if(method_choice = 'random'):
        best_params, best_score, history = run_optimization(pipeline, param_distributions, X_preprocessed, y, method='random', n_iter=10)
        print("Best Parameters:", best_params)
        print("Best Score:", best_score)
        plot_learning_curves(history, 'Random Search Learning Curve')
    else:
        optimizer = BayesianOptimization(pipeline, param_distributions, X, y, n_iter=10)
        best_params, best_score = optimizer.optimize()
        plot_learning_curves(optimizer.history, 'Bayesian Optimization Learning Curve')
    
    

if __name__ == "__main__":
    main()
