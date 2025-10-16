import numpy as np

def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_intercept(z_no_b: np.ndarray, target_rate: float, tol: float = 1e-5, max_iter: int = 50) -> float:
    """Find intercept b such that mean(sigmoid(z_no_b + b)) ~= target_rate."""
    lo, hi = -10.0, 10.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p = logistic(z_no_b + mid).mean()
        if p > target_rate:
            hi = mid
        else:
            lo = mid
        if abs(p - target_rate) < tol:
            return mid
    return (lo + hi) / 2
