import numpy as np
from scipy import stats
from .distances import euclidean_distances, manhattan_distances, cosine_distances


class KNN():
    def __init__(self, n_neighbors: int, distance_measure: str = "euclidean", aggregator: str = "mode"):
        self.n_neighbors = n_neighbors
        self.training_features = None
        self.training_targets = None

        if aggregator == "mode":
            self.aggregator = stats.mode
        elif aggregator == "mean":
            self.aggregator = np.mean
        elif aggregator == "median":
            self.aggregator = np.median
        else:
            raise ValueError(
                "aggregator must be either \"mode\", \"mean\", or \"median\"")

        if distance_measure == "euclidean":
            self.distance_function = euclidean_distances
        elif distance_measure == "manhattan":
            self.distance_function = manhattan_distances
        elif distance_measure == "cosine":
            self.distance_function = cosine_distances
        else:
            raise ValueError(
                "distance_measure must be either \"euclidean\", \"manhattan\", or \"cosine\"")

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.training_features = features
        self.training_targets = targets

    def predict(self, features: np.ndarray, ignore_first: bool = False) -> np.ndarray:
        nearest_indicies = np.argsort(self.distance_function(
            features, self.training_features), axis=1)
        nearest_indicies = (nearest_indicies[1:(self.n_neighbors + 1)]
                            if ignore_first
                            else nearest_indicies[:self.n_neighbors])

        predictions = np.ndarray(
            shape=(self.training_targets.shape[0], features.shape[1]))
        for sample_idx in range(0, predictions.shape[1]):
            predictions = self.aggregator(
                self.training_targets[:, nearest_indicies[sample_idx]], axis=1)

        return predictions