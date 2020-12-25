import numpy as np
from .knn import KNN

def collaborative_filter(features: np.ndarray, n_neighbors: int, distance_measure: str = "euclidean",
                         aggregator: str = "mode") -> np.ndarray:
    """
    Use the KNN model to impute missing values by looking at the n_neighbors closest vectors and
    aggregating over them.

    Parameters
    ----------
    features (np.ndarray) - features of shape (# features, # examples) which we want to impute the
     missing values of\n
    n_neighbors (int) - the number of closest vectors to consider\n
    distance_measure (str) - the distance measure to use when finding the closest vectors
     (\"euclidean\", \"manhattan\", or \"cosine\")\n
    aggregator (str) - the aggreagor to use over the closest vectors (\"mode\", \"median\", or \"mean\")
    """
    knn_model = KNN(n_neighbors, distance_measure, aggregator)
    knn_model.fit(features = features, targets = features)
    all_imputed = knn_model.predict(features, ignore_first=True)

    imputed_array = features
    for example_idx in range(0, features.shape[1]):
        for zero_idx in np.where(features[:, example_idx] == 0):
            imputed_array[zero_idx, example_idx] = all_imputed[zero_idx, example_idx]
    
    return imputed_array