import numpy as np


def simulate_brownian_motion(T, n_steps, n_mc, seed=1234):
    """Simulate standard Brownian motion paths.

    Parameters
    ----------
    T : float
        Time horizon.
    n_steps : int
        Number of time steps.
    n_mc : int
        Number of Monte Carlo paths.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tab_t : ndarray
        Time grid of shape (n_steps+1,).
    bm_paths : ndarray
        Brownian motion paths of shape (n_steps+1, n_mc).
    """
    np.random.seed(seed)
    dt = T / n_steps
    increments = np.random.normal(0, np.sqrt(dt), size=(n_steps, n_mc))
    tab_t = np.linspace(0, T, n_steps + 1)
    bm_paths = np.cumsum(increments, axis=0)
    bm_paths = np.insert(bm_paths, 0, 0, axis=0)  # insert B_0 = 0 at the beginning
    return tab_t, bm_paths


def simulate_brownian_bridge(a, b, t0, t1, n_steps, n_mc, seed=1234):
    """Simulate Brownian bridge paths from a to b over [t0, t1].

    Parameters
    ----------
    a, b : float
        Start and end values.
    t0, t1 : float
        Start and end times.
    n_steps : int
        Number of time steps.
    n_mc : int
        Number of Monte Carlo paths.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tab_t : ndarray
        Time grid of shape (n_steps+1,).
    bridge_paths : ndarray
        Brownian bridge paths of shape (n_steps+1, n_mc).
    """
    np.random.seed(seed)
    T = t1 - t0
    dt = T / n_steps
    increments = np.random.normal(0, np.sqrt(dt), size=(n_steps, n_mc))
    tab_t = np.linspace(t0, t1, n_steps + 1)
    bm_paths = np.cumsum(increments, axis=0)
    bm_paths = np.insert(bm_paths, 0, 0, axis=0)  # insert B_0 = 0 at the beginning
    # Adjust to create the bridge
    bridge_paths = bm_paths - (tab_t[:, None] - t0) / T * (bm_paths[-1, :] - (b - a))
    bridge_paths += a  # start at a
    return tab_t, bridge_paths
