from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(X):
    numeric_features = [0, 1, 2, 3]
    categorical_features = []

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numeric_features),
            ('num_scaler', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    return preprocessor.fit_transform(X)
