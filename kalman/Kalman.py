import numpy as np


class KF:
    def __init__(self, initial_x: float,
                 initial_v: float,
                 accel_variance: float,
                 dt: float) -> None:
        # initial state
        self._x = np.array((initial_x, initial_v))
        self._accel_variance = accel_variance
        self._dt = dt

        # internal matrices
        self._P = np.eye(2)
        self._F = np.eye(2)
        self._F[0, 1] = self._dt
        self._G = np.array((0.5 * self._dt ** 2, self._dt))

    def predict(self) -> None:
        """
        x = F x
        P = F P Ft + G Gt a
        """""
        self._x = self._F.dot(self._x)
        self._P = self._F.dot(self._P).dot(self._F.T) + self._G.dot(self._G.T) * self._accel_variance

    def update(self, meas_value: float, meas_variance: float):
        """
        y = z - H x - prediction error
        S = H P Ht + R - covariance error
        K = P Ht S^-1 - optimal Kalman step

        x = x + K y - updated location
        P = (I - K H) * P - updated covariance
        """""
        # TODO: refactor this function
        H = np.array((1, 0))
        z = np.array(meas_value)
        R = np.array(meas_variance)

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._P = new_P
        self._x = new_x

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self) -> float:
        return self._x[0]

    @property
    def vel(self) -> float:
        return self._x[1]
