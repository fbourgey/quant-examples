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


def generate_haar_functions(j:int, t):
    """Compute Haar functions at level j.

    Parameters
    ----------
    j : int
        Level parameter (creates 2^j functions).
    t : array_like
        Time points where functions are evaluated.

    Returns
    -------
    list of ndarray
        List of 2^j Haar functions H_{2^j+k} for k=0,...,2^j-1.
    """

    haar_functions = []

    for k in range(2**j):
        t_min = k / 2**j
        t_mid = (k + 0.5) / 2**j
        t_max = (k + 1) / 2**j
        haar_functions.append(
            2 ** (j / 2) * (t >= t_min) * (t < t_mid)
            - 2 ** (j / 2) * (t >= t_mid) * (t < t_max)
            + 0.0 * t
        )

    return haar_functions


def generate_schauder_functions(j, t):
    """Compute Schauder functions at level j.

    Parameters
    ----------
    j : int
        Level parameter (creates 2^j functions).
    t : array_like
        Time points where functions are evaluated.

    Returns
    -------
    list of ndarray
        List of 2^j Schauder functions S_{2^j+k} for k=0,...,2^j-1.
    """

    schauder_functions = []

    for k in range(2**j):
        t_min = k / 2**j
        t_mid = (k + 0.5) / 2**j
        t_max = (k + 1) / 2**j
        schauder_functions.append(
            2 ** (j / 2) * (t - t_min) * (t >= t_min) * (t < t_mid)
            + 2 ** (j / 2) * ((k + 1) / 2**j - t) * (t >= t_mid) * (t < t_max)
            + 0.0 * t
        )

    return schauder_functions


def simulate_brownian_levy_construction(t, L, seed=1234):
    """Simulate Brownian motion using Lévy-Ciesielski construction.

    Parameters
    ----------
    t : array_like
        Time points for evaluation.
    L : int
        Maximum level of Schauder expansion.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray
        Brownian motion values at times t.
    """
    np.random.seed(seed)
    N_L = 2 ** (L + 1)
    Z = np.random.normal(0, 1, N_L)
    t = np.array(np.atleast_1d(t))
    brownian_levy = t * Z[0]
    for j in range(L + 1):
        schauder_functions = generate_schauder_functions(j, t)
        for k in range(2**j):
            brownian_levy += schauder_functions[k] * Z[2**j + k]
    return brownian_levy
