import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, delta_t, sigma_a, sigma_x, sigma_y):
        self.delta_t = delta_t
        self.sigma_a = sigma_a
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        self.x_p = []
        self.y_p = []

        self.I = np.matrix(np.identity(6))

        # 状态变换矩阵
        self.F = np.matrix(np.array(
            [
                [1, self.delta_t, 1 / 2 * self.delta_t ** 2, 0, 0, 0],
                [0, 1, self.delta_t, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, self.delta_t, 1 / 2 * self.delta_t ** 2],
                [0, 0, 0, 0, 1, self.delta_t],
                [0, 0, 0, 0, 0, 1]
            ]
        ))

        # 过程噪声矩阵
        self.Q = np.matrix(np.array(
            [
                [self.delta_t**4 / 4, self.delta_t **
                    3 / 2, self.delta_t**2 / 2, 0, 0, 0],
                [self.delta_t**3 / 2, self.delta_t**2, self.delta_t, 0, 0, 0],
                [self.delta_t**2 / 2, self.delta_t**2, 1, 0, 0, 0],
                [0, 0, 0, self.delta_t**4 / 4, self.delta_t **
                    3 / 2, self.delta_t**2 / 2],
                [0, 0, 0, self.delta_t**3 / 2, self.delta_t**2, self.delta_t],
                [0, 0, 0, self.delta_t**2 / 2, self.delta_t, 1]
            ]
        )) * self.sigma_a**2

        # 测量不确定性矩阵
        self.R = np.matrix(np.array(
            [
                [self.sigma_y**2, 0],
                [0, self.sigma_y**2]
            ]))

        # 观测矩阵
        self.H = np.matrix(np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0]
            ]))

        self.x = np.matrix(np.zeros((6, 1)))

        self.P = np.matrix(np.zeros((6, 6)))
        self.P[0, 0] = self.P[1, 1] = self.P[2, 2] = 500
        self.P[3, 3] = self.P[4, 4] = self.P[5, 5] = 500

    def update(self, z):
        self.Kn = self.P.dot(self.H.transpose()).dot(
            (self.H.dot(self.P).dot(self.H.transpose()) + self.R).I)
        self.x = self.x + self.Kn.dot(z - self.H.dot(self.x))
        self.P = (self.I - self.Kn.dot(self.H)).dot(self.P)\
            .dot((self.I - self.Kn.dot(self.H)).transpose()) \
            + self.Kn.dot(self.R).dot(self.Kn.transpose())
        self.x_p.append(self.x[0, 0])
        self.y_p.append(self.x[3, 0])

    def predict(self):
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.transpose()) + self.Q


x_measured = [-393.66, -375.93, -351.04, -328.96, -299.35, -273.36, -245.89,
              -222.58, -198.03, -174.17, -146.32, -123.72, -103.47, -78.23, -52.63,
              -23.34, 25.96, 49.72, 76.94, 95.38, 119.83, 144.01, 161.84, 180.56, 201.42,
              222.62, 239.4, 252.51, 266.26, 271.75, 277.4, 294.12, 301.23, 291.8, 299.89]

y_measured = [300.4, 301.78, 295.1, 305.19, 301.06, 302.05, 300, 303.57, 296.33,
              297.65, 297.41, 299.61, 299.6, 302.39, 295.04, 300.09, 294.72, 298.61,
              294.64, 284.88, 272.82, 264.93, 251.46, 241.27, 222.98, 203.73, 184.1,
              166.12, 138.71, 119.71, 100.41, 79.76, 50.62, 32.99, 2.14]


kf = KalmanFilter(1, 0.5, 3, 3)
# 初始化后先预测n->n+1
kf.predict()

for i in range(len(x_measured)):
    z = np.matrix(np.array([[x_measured[i]], [y_measured[i]]]))
    kf.update(z)
    kf.predict()


# true
x_true = np.linspace(-400, 300, 1000)
y_true = []
for i in x_true:
    if i < 0:
        y_true.append(300)
    else:
        y_true.append(((300)**2 - i**2)**(1 / 2))

# x_2_true = np.


plt.plot(kf.x_p, kf.y_p, 'ro-', label='Estimates')
plt.plot(x_measured, y_measured, 'bv-', label='Measured')
plt.plot(x_true, y_true, 'g-', label='True')
plt.legend(loc='best')
plt.xlabel("X(m)")
plt.ylabel("Y(m)")
plt.title("Kalman filter for vehicle estimates")
plt.show()
