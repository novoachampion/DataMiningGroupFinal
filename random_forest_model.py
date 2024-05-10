from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def train_and_evaluate_rf(X, y, test_size=0.2, random_state=42, n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=4):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the Random Forest Regressor with parameters to reduce overfitting
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Convert predictions and targets to binary classes for classification metrics
    y_pred_class = (y_pred > y_pred.mean(axis=0)).astype(int)
    y_test_class = (y_test > y_test.mean(axis=0)).astype(int)

    precision = precision_score(y_test_class, y_pred_class, average='macro')
    recall = recall_score(y_test_class, y_pred_class, average='macro')
    f1 = f1_score(y_test_class, y_pred_class, average='macro')

    # Print evaluation metrics
    print("Random Forest Model Evaluation:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared score: {r2}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return model, y_test, y_pred, precision, recall, f1
