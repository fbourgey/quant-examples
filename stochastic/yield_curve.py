import numpy as np


def nelson_siegel(tau, beta0, beta1, beta2, lbd):
    """
    Nelson-Siegel yield curve model.

    Parameters
    ----------
    tau : float or array_like
        Time to maturity.
    beta0 : float
        Long-term level parameter.
    beta1 : float
        Short-term component parameter.
    beta2 : float
        Medium-term component parameter.
    lbd : float
        Decay parameter (lambda).

    Returns
    -------
    float or ndarray
        Yield curve values.
    """
    exp_term = (1.0 - np.exp(-lbd * tau)) / (lbd * tau)
    return beta0 + beta1 * exp_term + beta2 * (exp_term - np.exp(-lbd * tau))


def nelson_siegel_loadings(tau, lbd):
    """
    Compute the Nelson-Siegel factor loadings.

    Parameters
    ----------
    tau : array_like
        Time to maturity.
    lbd : float
        Decay parameter (lambda).

    Returns
    -------
    ndarray
        Factor loadings matrix of shape (len(tau), 3).
    """
    tau = np.asarray(tau)
    exp_term = (1.0 - np.exp(-lbd * tau)) / (lbd * tau)
    loading0 = np.ones_like(tau)
    loading1 = exp_term
    loading2 = exp_term - np.exp(-lbd * tau)
    return np.vstack([loading0, loading1, loading2]).T
