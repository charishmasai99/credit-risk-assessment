import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Drop ID column
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)

    target_column = "Status"

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns.tolist()
