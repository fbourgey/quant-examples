# quant-examples

A collection of examples and tutorials on quantitative finance,
numerical methods, stochastic processes, and simulation.

## Contents (Jupyter notebooks)

- [brownian_bridge.ipynb](./brownian_bridge.ipynb): Brownian Bridge construction and applications.
- [cos_method.ipynb](./cos_method.ipynb): COS method for option pricing.
- [gaussian_pca.ipynb](./gaussian_pca.ipynb): Principal Component Analysis of two-dimensional Gaussian data.
- [fractional_brownian_motion.ipynb](./fractional_brownian_motion.ipynb): Fractional Brownian Motion construction.
- [levy_construction_brownian_motion.ipynb](./levy_construction_brownian_motion.ipynb): Lévy's construction of Brownian Motion.

## Python Installation Guide

### Option 1: Standard Virtual Environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/fbourgey/quant-examples.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd quant-examples
   ```

3. **Create a virtual environment:**

   ```bash
   python3 -m venv .venv
   ```

4. **Activate the environment:**

   ```bash
   source .venv/bin/activate
   ```

5. **Install dependencies:**

   ```bash
   pip install .
   ```

6. **Launch Jupyter Lab (optional):**

   ```bash
   jupyter lab
   ```

---

### Option 2: Using `uv` (Recommended)

If you have [`uv`](https://docs.astral.sh/uv/) installed, setup is simpler and faster.  
After cloning the repository (steps 1–2 above), run:

```bash
uv sync
```

This will automatically create a virtual environment and install all dependencies.
