import numpy as np

def euclidean_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise euclidean distances between the examples of the two matricies 

    Parameters
    ----------
    X (np.ndarray) - matrix #1 of shape (x # features, n # examples)\n
    Y (np.ndarray) - matrix #2 of shape (x # features, m # examples)

    Output
    ------
    distances (np.ndarray) - a matrix of shape (n, m) containing all of the pairwise distances
    """
    x = X.shape[0]
    n = X.shape[1]
    m = Y.shape[1]

    X_modified = X.reshape((x, n, 1))
    Y_modified = Y.reshape((x, 1, m))

    return np.linalg.norm(X_modified - Y_modified, ord = 2, axis = 0)

def manhattan_distances(X : np.ndarray, Y : np.ndarray) -> np.ndarray:
    """
    Computes the pairwise Manhattan distances between the examples of the two matricies 

    Parameters
    ----------
    X (np.ndarray) - matrix #1 of shape (x # features, n # examples)\n
    Y (np.ndarray) - matrix #2 of shape (x # features, m # examples)

    Output
    ------
    distances (np.ndarray) - a matrix of shape (n, m) containing all of the pairwise distances
    """
    x = X.shape[0]
    n = X.shape[1]
    m = Y.shape[1]

    X_modified = X.reshape((x, n, 1))
    Y_modified = Y.reshape((x, 1, m))

    return np.sum(np.abs(X_modified - Y_modified), axis = 0)

def cosine_distances(X : np.ndarray, Y : np.ndarray) -> np.ndarray:
    """
    Computes the pairwise cosine distances between the examples of the two matricies 

    Parameters
    ----------
    X (np.ndarray) - matrix #1 of shape (x # features, n # examples)\n
    Y (np.ndarray) - matrix #2 of shape (x # features, m # examples)

    Output
    ------
    distances (np.ndarray) - a matrix of shape (n, m) containing all of the pairwise distances
    """
    distances = np.ndarray(shape = (X.shape[1], Y.shape[1]))
    for x_idx in range(0, X.shape[1]):
        for y_idx in range(0, Y.shape[1]):
            x_vec = X[:, x_idx].flatten()
            y_vec = Y[:, y_idx].flatten()

            distances[x_idx, y_idx] = 1 - np.dot(x_vec, y_vec)/(np.linalg.norm(x_vec)*np.linalg.norm(y_vec))
    
    return distances