import numpy as np
from scipy import stats


def gaussian_mixture_pdf(x, weights, means, variances):
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
    variances : array_like, shape (n_components, n_features)
        Variances of each Gaussian component.

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
    means = np.atleast_1d(np.asarray(means))
    variances = np.atleast_1d(np.asarray(variances))
    n_components = len(weights)

    if weights.sum() != 1:
        raise ValueError("Weights must sum to 1.")

    if means.shape[0] != n_components:
        raise ValueError("Means must be of shape (n_components,).")

    if variances.shape[0] != n_components:
        raise ValueError("Variances must be of shape (n_components,).")

    pdf_values = np.zeros(x.shape[0])

    for i in range(n_components):
        pdf_values += weights[i] * stats.norm.pdf(
            x, loc=means[i], scale=np.sqrt(variances[i])
        )

    return pdf_values
