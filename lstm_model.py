import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_and_evaluate_lstm(X, y, test_size=0.2, random_state=42, epochs=50, batch_size=16):
    X = X.values.reshape((X.shape[0], X.shape[1], 1))
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    y_pred_class = (y_pred > y_pred.mean(axis=0)).astype(int)
    y_test_class = (y_test > y_test.mean(axis=0)).astype(int)

    precision = precision_score(y_test_class, y_pred_class, average='macro')
    recall = recall_score(y_test_class, y_pred_class, average='macro')
    f1 = f1_score(y_test_class, y_pred_class, average='macro')

    print("LSTM Model Evaluation:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared score: {r2}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return model, y_test, y_pred, precision, recall, f1
