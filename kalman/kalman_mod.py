import numpy as np
import numba as nb
from kalman_base import*


@nb.njit()
def predict_smooth(state: np.ndarray,
                   P: np.ndarray,
                   dt: float,
                   accel_variance: float):
    """
    Hidden transition model: 
    
    X_{k+1} = X_{k} + V_k * dt + a * dt^2 / 2
    V_{k+1} = V_{k} + a * dt
    """""

    F = np.array(((1, dt),
                  (0, 1)))
    G = np.array((0.5 * dt ** 2, dt))
    Q = G.dot(G.T) * accel_variance

    return predict(state, P, F, Q)


@nb.njit()
def update_smooth(state: np.ndarray,
           P: np.ndarray,
           meas_value: float,
           meas_variance: float):
    """
    Observation model:
    
    Z = H * X + e
    H = [1  0]
    """""

    z = np.array(meas_value)
    H = np.array((1, 0))
    R = np.array(meas_variance)

    return update(state, P, z, H, R)

# TODO: make smooth_batch
# TODO: make cointegration_filtering



