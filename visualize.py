import matplotlib.pyplot as plt

def plot_predictions(y_test, predictions, title):
    plt.scatter(y_test.iloc[:, 0], predictions[:, 0], alpha=0.5)
    plt.title(f'{title}: True vs Predicted')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()
