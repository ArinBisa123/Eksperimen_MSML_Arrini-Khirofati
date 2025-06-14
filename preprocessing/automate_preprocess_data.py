import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
# from sklearn.utils import resample, shuffle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# import joblib


def preprocessing_pipeline(csv_path):
    dataset = pd.read_csv(csv_path)
    numeric_cols = ['Age', 'Sex', 'Settlement size']
    categorical_cols = dataset.select_dtypes(include='object')
    categorical_features = categorical_cols.columns.to_list()
    # 1. Drop fitur yang tidak digunakan
    dataset = dataset.drop(columns=['ID'], inplace=True, axis=1)

    # 2. Menangani Outliers
    Q1 = dataset[numeric_cols].quantile(0.25)
    Q3 = dataset[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    filter_outliers = ~((dataset[numeric_cols] < (Q1 - 1.5 * IQR)) |
                        (dataset[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    dataset = dataset[filter_outliers]

    # 3. Split data menjadi train dan test
    X = dataset.drop(columns=['Income'])
    y = dataset['Income']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.1,
                                                        random_state=123)
    # 4. Encoding & Scaling
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    return X_train_enc, X_test_enc, y_train, y_test, preprocessor


file_path = f'./sgdata.csv'
data_final = preprocessing_pipeline(file_path)
data_final.to_csv("sgdata_preprocessed.csv")
