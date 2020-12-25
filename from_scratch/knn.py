import numpy as np
from scipy import stats
from .distances import euclidean_distances, manhattan_distances, cosine_distances

class KNN():
    """
    A basic K Nearest Neighbors classifier which uses euclidean, manhattan, or cosine distance.

    Member Variables
    ----------------
    n_neighbors (int) - the number of neighbors considered when predicting values\n
    training_features (np.ndarray) - the features used in training of shape (# features, # examples)\n
    training_targets (np.ndarray) - the targets used in training of shape (# dimensions, # examples)\n
    aggregator (function) - the aggregate used to predict values (mode, mean, or median)\n
    distance_function (function) - the function used to calculate distances between vectors
     (euclidean, manhattan, or cosine)
    """

    def __init__(self, n_neighbors: int, distance_measure: str = "euclidean", aggregator: str = "mode"):
        """
        Initialize the member variables.\n
        distance_measure must take the value of either \"euclidean\", \"manhattan\", or \"cosine\".\n
        aggregator must take the value of either \"mode\", \"median\", or \"mean\".\n
        """
        self.n_neighbors = n_neighbors
        self.training_features = None
        self.training_targets = None

        # Initialize self.aggregator with mapping str -> function
        if aggregator == "mode":
            self.aggregator = (lambda a, axis: stats.mode(a, axis)[0])
        elif aggregator == "mean":
            self.aggregator = (lambda a, axis: np.mean(a, axis))
        elif aggregator == "median":
            self.aggregator = (lambda a, axis: np.median(a, axis))
        else:
            raise ValueError(
                "aggregator must be either \"mode\", \"mean\", or \"median\"")

        # Initializing self.distance_function with mapping str -> function
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
        """
        Lazily fit the KNN model to the training data by storing the features and targets for
        use during the self.predict function

        Parameters
        ----------
        features (np.ndarray) - the features used in training of shape (# features, # examples)\n
        targets (np.ndarray) - the targets used in training of shape (# dimensions, # examples)
        """
        self.training_features = features
        self.training_targets = targets

    def predict(self, features: np.ndarray, ignore_first: bool = False) -> np.ndarray:
        """
        Predict the values of each vector in features by aggregating the targets corresponding to the
        self.n_neighbors closest vectors in self.training_features. If ignore_first == True, then we suppose
        that this KNN classifier is being used as a collaborative filter and ignore the closest vector, which
        in this case is one's self.  

        Parameters
        ----------
        features (np.ndarray) - the vectors we want to predict the values of shape (# features, # examples)\n
        ignore_first (bool) - a flag which indicates if we ignore the closes vector.  This should be true in the
            case of a collaborative filter.
        """
        nearest_indicies = np.argsort(self.distance_function(
            features, self.training_features), axis=1)
        nearest_indicies = (nearest_indicies[1:(self.n_neighbors + 1)]
                            if ignore_first
                            else nearest_indicies[:, 0:self.n_neighbors])

        predictions = np.ndarray(shape=(
            self.training_targets.shape[0], features.shape[1]), dtype=self.training_targets.dtype)
        for sample_idx in range(0, predictions.shape[1]):
            aggregate = self.aggregator(
                self.training_targets[:, nearest_indicies[sample_idx]], axis=1)
            predictions[:, sample_idx] = aggregate
        return predictions
