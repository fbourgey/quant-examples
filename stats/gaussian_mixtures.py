import numpy as np
from scipy.stats import multivariate_normal


def gaussian_mixture_pdf(x, weights, means, covariances):
    """
    Compute the probability density function of a Gaussian Mixture Model.

    Parameters
    ----------
    x : array_like, shape (n_samples, n_features)
        Points where the PDF is evaluated.
    weights : array_like, shape (n_components,)
        Weights of each Gaussian component. Must sum to 1.
    means : array_like, shape (n_components, n_features)
        Means of each Gaussian component.
    covariances : array_like, shape (n_components, n_features, n_features)
        Covariance matrices of each Gaussian component.

    Returns
    -------
    ndarray, shape (n_samples,)
        The computed PDF values at each point in x.

    Raises
    ------
    ValueError
        If weights don't sum to 1, or if means/covariances have incorrect shapes.
    """
    x = np.atleast_1d(np.asarray(x))
    weights = np.atleast_1d(np.asarray(weights))
    means = np.atleast_2d(np.asarray(means))
    covariances = np.atleast_3d(np.asarray(covariances))

    if weights.sum() != 1:
        raise ValueError("Weights must sum to 1.")

    if means.ndim != 2 or means.shape[0] != weights.shape[0]:
        raise ValueError(
            "Means must be a 2D array with shape (n_components, n_features)."
        )

    if covariances.ndim != 3 or covariances.shape[0] != weights.shape[0]:
        raise ValueError(
            "Covariances must be a 3D array with shape "
            "(n_components, n_features, n_features)."
        )

    n_components = len(weights)
    pdf_values = np.zeros(x.shape[0])

    for i in range(n_components):
        pdf_values += weights[i] * multivariate_normal.pdf(
            x, mean=means[i], cov=covariances[i]
        )

    return pdf_values
