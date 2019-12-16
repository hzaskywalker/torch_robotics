import numpy as np


def pca(X, d):

    u, s, vh = np.linalg.svd(X, full_matrices=True)
    #print(u.shape, s.shape, vh.shape)
    return np.dot(X, vh.T)
    # Data matrix X, assumes 0-centered
    # normalized
    X = X.astype('float64')
    n, m = X.shape
    #X = X - X.mean(axis=0)[None, :]
    #assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    #C = np.dot(X.T, X) / (n-1)
    C = np.cov(X.T, rowvar=False)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eigh(C)
    idx = np.argsort(eigen_vals)[::-1][:d]
    eigen_vecs = eigen_vecs[:, idx]
    eigen_vals = eigen_vals[idx]
    #return np.dot(X, eigen_vecs)
    #out = np.dot(X, eigen_vecs)
    return eigen_vecs