import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.interpolate import RegularGridInterpolator
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd
import seaborn as sns
from scipy.stats import norm

###############################################################################

# Define auxiliary functions for Bayesian Optimization

##############################################################################

def generate_oil_field_gauss(grid_size=2000, resolution=10,
                             max_yield=20000, angle=20.0):
    """
    Generates a 2D grid representing an oil field with a Gaussian-distributed
    reservoir that can be rotated.

    Args:
        grid_size (int): size of the grid in km (default 2000×2000).
        resolution (int): step size in km.
        max_yield (int): peak oil yield in barrels/day.
        angle (float): rotation of the reservoir in degrees (counter‑clockwise).

    Returns:
        X, Y: meshgrid coordinates.
        Z: 2D array of oil yields.
    """
    # coordinate grid
    x = np.arange(0, grid_size, resolution)
    y = np.arange(0, grid_size, resolution)
    X, Y = np.meshgrid(x, y)

    # reservoir centre
    center_x, center_y = 1200, 600

    # spread
    sigma_x, sigma_y = 150, 500

    # rotate coordinates about the centre
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    Xc = X - center_x
    Yc = Y - center_y
    Xr = cos_t * Xc + sin_t * Yc
    Yr = -sin_t * Xc + cos_t * Yc

    # gaussian on rotated coords
    Z = max_yield * np.exp(
        -((Xr**2) / (2 * sigma_x**2) + (Yr**2) / (2 * sigma_y**2))
    )

    return X, Y, Z

############################################################################3


def drill_for_oil_interpolated(X_coords, X_grid, Y_grid, Z_grid):
    """
    Simulates drilling for oil by interpolating from pre-computed 2D grids.
    
    Args:
        X_coords: A 2D numpy array of shape (n_samples, 2) containing [x, y] coordinates to drill.
        X_grid: 2D meshgrid of X coordinates.
        Y_grid: 2D meshgrid of Y coordinates.
        Z_grid: 2D array of oil yields.
        
    Returns:
        1D array of interpolated oil yields for the requested coordinates.
    """
    # 1. Extract the 1D coordinate arrays from the 2D meshgrids
    # meshgrid repeats the 1D arrays, so we just grab the first row/column
    x_1d = X_grid[0, :]
    y_1d = Y_grid[:, 0]
    
    # 2. Set up the interpolator
    # Note: np.meshgrid(indexing='xy') creates Z_grid with shape (len(y), len(x)).
    # Therefore, the interpolator expects the axes in (Y, X) order.
    # We use bounds_error=False and fill_value=0 so if the algorithm tries to 
    # drill outside our 2000x2000 map, it just returns 0 barrels.
    interpolator = RegularGridInterpolator(
        (y_1d, x_1d), 
        Z_grid, 
        bounds_error=False, 
        fill_value=0
    )
    
    # 3. Format the query points
    # The algorithm provides [x, y], but the interpolator needs [y, x] to match the grid shape
    query_points = np.column_stack((X_coords[:, 1], X_coords[:, 0]))

    # 4. Return the interpolated yields
    yields = interpolator(query_points)
    
    # 5. Add small amount of gaussian noise at the end
    noise = np.random.normal(0, 100, size=yields.shape[0])  # mean=0, std=100
    noisy_yields = yields + noise

    # if negative, change to zero
    noisy_yields = np.maximum(noisy_yields, 0)

    return noisy_yields

##################################################################################################

def expected_improvement(x_query, gp, best_yield, xi=0.01):
    """
    Calculates the Expected Improvement at given points.
    
    Args:
        x_query: A 2D array of coordinates where we want to evaluate EI.
        gp: The trained GaussianProcessRegressor model.
        best_yield: The maximum oil yield observed in our actual drills so far.
        xi: The exploration parameter (default 0.01).
        
    Returns:
        1D array of Expected Improvement values for each query point.
    """
    # 1. Ask the GP for its predictions and uncertainty at the query points
    mu, sigma = gp.predict(x_query, return_std=True)
    
    # Flatten arrays to ensure the math operations broadcast correctly
    mu = mu.reshape(-1)
    sigma = sigma.reshape(-1)
    
    # 2. Safety first: Avoid division by zero
    # If sigma is exactly 0, we've drilled there before. We set a tiny floor.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 3. Calculate Z
    Z = (mu - best_yield - xi) / sigma_safe
    
    # 4. Calculate Expected Improvement
    ei = (mu - best_yield - xi) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # If uncertainty is effectively zero, EI should be zero
    ei[sigma == 0.0] = 0.0
    
    return ei


##################################################################################################



def plot_bo_step_2d(X_search_grid, Y_search_grid, gp, ei_values, current_drills, next_drill_site, step_number):
    """Plots the 2D GP Mean and Expected Improvement at a specific step."""
    
    # 1. Get the GP's predicted mean for the current step
    search_coords = np.vstack((X_search_grid.flatten(), Y_search_grid.flatten())).T
    mu, _ = gp.predict(search_coords, return_std=True)
    
    # 2. Reshape 1D arrays back to 2D grids for contour plotting
    mu_2d = mu.reshape(X_search_grid.shape)
    ei_2d = ei_values.reshape(X_search_grid.shape)
    
    # 3. Setup the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Panel 1: The GP Predicted Mean ---
    c1 = ax1.contourf(X_search_grid, Y_search_grid, mu_2d, levels=50, cmap='magma')
    # Plot past drills (Black dots)
    ax1.scatter(current_drills[:, 0], current_drills[:, 1], c='white', edgecolors='black', marker='o', s=50, label='Past Drills')
    # Plot the newly chosen drill site (Red Star)
    ax1.scatter(next_drill_site[0], next_drill_site[1], c='red', marker='*', s=200, edgecolors='black', label='Targeting Next')
    
    ax1.set_title(f"Step {step_number}: Surrogate Predicted Mean")
    fig.colorbar(c1, ax=ax1)
    ax1.legend(loc="upper left")
    
    # --- Panel 2: The Expected Improvement (Acquisition) ---
    c2 = ax2.contourf(X_search_grid, Y_search_grid, ei_2d, levels=50, cmap='viridis')
    # Highlight the peak of EI where the drill is going
    ax2.scatter(next_drill_site[0], next_drill_site[1], c='red', marker='*', s=200, edgecolors='black')
    
    ax2.set_title("Acquisition: Expected Improvement")
    fig.colorbar(c2, ax=ax2)
    
    plt.tight_layout()
    plt.show()