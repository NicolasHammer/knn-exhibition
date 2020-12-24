import numpy as np
from .knn import KNN

def collaborative_filter(features: np.ndarray, n_neighbors: int, distance_measure: str = "euclidean",
                         aggregator: str = "mode") -> np.ndarray:
    knn_model = KNN(n_neighbors, distance_measure, aggregator)
    knn_model.fit(features = features, targets = features)
    all_imputed = knn_model.predict(features, ignore_first=True)

    imputed_array = features
    for example_idx in range(0, features.shape[1]):
        for zero_idx in np.where(features[:, example_idx] == 0):
            imputed_array[zero_idx, example_idx] = all_imputed[zero_idx, example_idx]
    
    return imputed_array