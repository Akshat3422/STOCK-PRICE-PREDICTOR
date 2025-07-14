from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_val, X_test, columns):
    scalers = {}
    for col in columns:
        scaler = StandardScaler()
        X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
        X_val[col] = scaler.transform(X_val[col].values.reshape(-1, 1))
        X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
        scalers[col] = scaler
    return X_train, X_val, X_test, scalers
