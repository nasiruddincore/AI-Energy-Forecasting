from sklearn.ensemble import RandomForestRegressor

def train_model(train):
    X = train[['hour', 'day', 'month']]
    y = train['energy']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    print("Model trained successfully")

    return model

def predict(model, test):
    X_test = test[['hour', 'day', 'month']]
    predictions = model.predict(X_test)

    return predictions