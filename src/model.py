import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_xgboost_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test
