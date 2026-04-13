import matplotlib.pyplot as plt

def plot_results(actual, predicted, save_path):
    plt.figure(figsize=(10, 5))

    plt.plot(actual.values, label="Actual", marker='o')
    plt.plot(predicted, label="Predicted", linestyle='--')

    plt.title("Energy Consumption Forecast")
    plt.xlabel("Time Index")
    plt.ylabel("Energy")
    plt.legend()

    plt.savefig(save_path)
    plt.show()

    print(f"Graph saved at: {save_path}")