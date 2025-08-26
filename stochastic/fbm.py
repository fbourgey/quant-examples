# Simulate fractional Brownian motion (fBm) paths
import matplotlib as mpl
import numpy as np
import seaborn as sns
from scipy.special import hyp2f1

sns.set_theme("talk")
mpl.rcParams["figure.figsize"] = (8, 6)


def covariance_levy_fbm(u, v, H):
    r"""
    Compute the covariance matrix of Levy's fractional Brownian motion.

    It corresponds to:

    Cov(W_u^H, W_v^H) = 2 * H * \int_0^{min(u,v)} (u-s)^{H-1/2} (v-s)^{H-1/2} ds

    where:

    W_u^H = \sqrt{2H} * \int_0^u (u-s)^{H-1/2} dW_s

    Observe that W_u^H is normal with zero mean and variance u^{2H}.

    Parameters
    ----------
    u : np.ndarray or float
        First set of time points.
    v : np.ndarray or float
        Second set of time points.
    H : float
        Hurst parameter.

    Returns
    -------
    np.ndarray
        Covariance matrix evaluated at (u, v).
    """
    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    u_max_v = np.maximum(u, v)
    u_min_v = np.minimum(u, v)
    cov = (
        u_min_v ** (H + 0.5)
        * u_max_v ** (H - 0.5)
        * hyp2f1(1.0, 0.5 - H, 1.5 + H, u_min_v / u_max_v)
    )
    cov *= (2 * H) / (H + 0.5)
    return cov


def simulate_fbm(t, H: float, n_steps: int, n_paths: int, seed=None):
    """
    Simulate sample paths of fractional Brownian motion (fBm) via
    Cholesky decomposition.

    Parameters
    ----------
    t : float
        Final time of the simulation interval [0, t].
    H : float
        Hurst parameter (0 < H < 1) controlling the roughness of the paths.
    n_steps : int
        Number of time steps (the path will have n_steps + 1 points including 0).
    n_paths : int
        Number of independent fBm sample paths to simulate.
    seed : int or None, optional
        Random seed for reproducibility. If None, the random number generator is
        not seeded.

    Returns
    -------
    tab_t : np.ndarray, shape (n_steps + 1,)
        Array of time points at which the fBm is evaluated.
    fbm_paths : np.ndarray, shape (n_steps + 1, n_paths)
        Simulated fBm paths. Each column corresponds to a sample path.
    """
    if seed is not None:
        np.random.seed(seed)

    tab_t = np.linspace(0, t, n_steps + 1)
    u = np.tile(tab_t[1:], (n_steps, 1)).T
    z = np.random.normal(size=(n_steps, n_paths))
    L = np.linalg.cholesky(covariance_levy_fbm(u, u.T, H))
    fbm_paths = L @ z
    fbm_paths = np.insert(fbm_paths, 0, 0, axis=0)

    return tab_t, fbm_paths
