import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, z, delta_t, sigma_a, sigma_x, sigma_y):
        self.z = z
        self.delta_t = delta_t
        self.sigma_a = sigma
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        # 状态变换矩阵
        self.F = np.array(
            [
                [1, self.delta_t, 1 / 2 * self.delta_t ^ 2, 0, 0, 0]
                [0, 1, self.delta_t, 0, 0, 0]
                [0, 0, 1, 0, 0, 0]
                [0, 0, 0, 1, self.delta_t, 1 / 2 * self.delta_t ^ 2]
                [0, 0, 0, 0, 0, 1]
            ]
        )

        self.Q = np.array(
            [

            ]
        )

    def update(self):
        pass

    def predict(self):


x = np.array()
