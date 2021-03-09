import pdb

import numpy as np
from matplotlib import pyplot as plt


def plot_tournament_history(data_path="results_history.npy"):
    tournament_data = np.load(data_path)
    num_generation, num_candidate = tournament_data.shape
    all_points = []
    average_performance = []
    for i in range(num_generation):
        candidates = tournament_data[i]
        valid_performances = []
        for j in range(num_candidate):
            if candidates[j] > 0:
                all_points.append((i + 1, tournament_data[i, j]))
                valid_performances.append(tournament_data[i, j])
        average_performance.append(np.mean(valid_performances))
    all_x = [point[0] for point in all_points]
    all_y = [point[1] for point in all_points]
    plt.scatter(all_x, all_y, alpha=0.3, label="candidates")
    x_generations = list(range(1, num_generation + 1))
    plt.plot(x_generations, average_performance, 'r', label="average")
    plt.xlabel("Generations")
    plt.ylabel("Cifar10 Accuracy")
    plt.title("Regularized Evolution Architecture search with 10-layer Deep Networks")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_tournament_history()
