import numpy as np
import numba as nb


@nb.njit()
def predict(state: np.ndarray, dt: float, accel_variance: float, P: np.ndarray):
    """
    x = F x
    P = F P Ft + G Gt a
    """""
    # P = np.eye(2)
    F = np.array(((1, dt),
                  (0, 1)))
    G = np.array((0.5 * dt ** 2, dt))

    state = F.dot(state)
    P = F.dot(P).dot(F.T) + G.dot(G.T) * accel_variance
    return state, P


@nb.njit()
def update(state: np.ndarray, meas_value: float, meas_variance: float, P: np.ndarray, H: np.ndarray):
    """
    y = z - H x - prediction error
    S = H P Ht + R - covariance error
    K = P Ht S^-1 - optimal Kalman step

    x = x + K y - updated location
    P = (I - K H) * P - updated covariance
    """""

    # H = np.array((1, 0))
    z = np.array(meas_value)
    R = np.array(meas_variance)

    y = z - H.dot(state)
    S = H.dot(P).dot(H.T) + R

    K = P.dot(H.T).dot(np.linalg.inv(S))

    state = state + K.dot(y)
    P = (np.eye(2) - K.dot(H)).dot(P)
    return state, P


@nb.njit()
def filter_batch(data: np.ndarray, data_variance: np.ndarray, init_state: np.ndarray, init_P: np.ndarray, H: np.ndarray):
    x = np.zeros_like(data+1)
    v = np.zeros_like(data+1)
    x[0] = init_state[0]
    v[0] = init_state[1]

    state = init_state
    P = init_P
    for i in range(len(data)):
        state, P = update(state=state,
                          meas_value=data[i],
                          meas_variance=data_variance[i],
                          P=P,
                          H=H)

        x[i+1] = state[0]
        v[i+1] = state[1]

    return x, v
