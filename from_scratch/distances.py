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

    X_modified = np.reshape(X, (x, n, 1))
    Y_modified = Y.reshape(Y, (x, 1, m))

    return np.sum(np.abs(X_modified - Y_modified))

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
    return 1 - np.matmul(X.T, Y)/(np.linalg.norm(X, ord = 2, axis = 1)
                                  *np.linalg.norm(Y, ord = 2, axis = 1))