from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print("\nModel Evaluation:")
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))