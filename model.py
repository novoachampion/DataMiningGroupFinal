from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # For classification metrics, we need to binarize or threshold the predictions
    y_pred_class = (y_pred > np.median(y_pred)).astype(int)
    y_test_class = (y_test > np.median(y_test)).astype(int)

    precision = precision_score(y_test_class, y_pred_class, average='macro')
    recall = recall_score(y_test_class, y_pred_class, average='macro')
    f1 = f1_score(y_test_class, y_pred_class, average='macro')

    print("Model Evaluation:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared score: {r2}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return model, y_test, y_pred, precision, recall, f1
