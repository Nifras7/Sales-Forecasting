import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return predictions, rmse

def plot_forecast(y_test, predictions, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual Sales', color='red')
    plt.plot(y_test.index, predictions, label='Predicted Sales', color='green')
    plt.title('Sales Forecasting using XGBoost')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
