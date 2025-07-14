import joblib
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import mean_squared_error
from feature_engineering import scale_features
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from preprocess import preprocess_stock_data
def train_price_model(df):
    
    models = {
        'Lasso': {
            'model': Lasso(),
            'params': {'alpha': [0.1, 1.0, 10.0]}
        },
        'Ridge': {
            'model': Ridge(),
            'params': {'alpha': [0.1, 1.0, 10.0]}
        },
        'Xgboost': {
            'model': XGBRegressor(),
            'params': {'n_estimators': [100,200], 'learning_rate': [0.01,0.1]}
        },
        'RandomForest': {
            'model': RandomForestRegressor(),
            'params': {'n_estimators': [100,200], 'max_depth': [10,20]}
        },
        'SVR': {
            'model': SVR(),
            'params': {'kernel': ['rbf'], 'C': [1.0], 'gamma':['scale']}
        },
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {}
        }
    }

    top10 = ['Close_lag1', 'Return_lag1',
         'RSI_14', 'Volatility_14',
         'Volume_lag1','OBV','price_sentiment']
    features_df = df[top10].dropna()
    target = df['Close'].loc[features_df.index]     
    # Split the data into training, validation, and test sets
    n=len(features_df)
    train_size = int(n * 0.7)
    val_size   = int(n * 0.15)

    X_train = features_df.iloc[:train_size]
    y_train = target.iloc[:train_size]

    X_val   = features_df.iloc[train_size:train_size+val_size]
    y_val   = target.iloc[train_size:train_size+val_size]

    X_test  = features_df.iloc[train_size+val_size:]
    y_test  = target.iloc[train_size+val_size:]

    columns_to_scale =[f for f in top10 if f != 'price_sentiment']

    X_train, X_val, X_test, scalers= scale_features(X_train, X_val, X_test, columns_to_scale)

    models_results = {}
    for model_name, model_info in models.items():
        try:
            model = model_info['model']
            params = model_info['params']
            if params:
            # Use RandomizedSearchCV if the model has hyperparameters to tune

                clf=GridSearchCV(model, param_grid=params, cv=3,scoring='neg_mean_squared_error', n_jobs=-1)
                # Fit the model
                clf.fit(X_train, y_train)
                best_model = clf.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model
            # Evaluate the model on the validation set
            val_score = best_model.score(X_val, y_val)
            val_loss = mean_squared_error(y_val, best_model.predict(X_val))
            test_score = best_model.score(X_test, y_test)
            test_loss = mean_squared_error(y_test, best_model.predict(X_test))

            print(f"{model_name} validation score: {val_score:.4f}, test score: {test_score:.4f}")

            
            # Store the results
            models_results[model_name] = {
            'model': best_model,
            'val_score': val_score,
            'val_loss': val_loss,
            'test_score': test_score,
            'test_loss': test_loss
        }
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue

    # Save the best model
    best_model_name = min(models_results, key=lambda x: models_results[x]['val_loss'])
    best_model = models_results[best_model_name]['model']

    print(f"\n✅ Best model: {best_model_name}")
    print(f"Validation Loss: {models_results[best_model_name]['val_loss']:.4f}")
    print(f"Test Loss: {models_results[best_model_name]['test_loss']:.4f}")
    print(f"Test Score: {models_results[best_model_name]['test_score']:.4f}")
    # Save the best model, scalers, and features
    joblib.dump({'model': best_model, 'scalers': scalers, 'features': top10}, 'best_model_bundle.pkl')
    return models_results[best_model_name]['test_score']

