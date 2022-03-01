import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, delta_t, sigma_a, sigma_x):
        self.delta_t = delta_t
        self.sigma_a = sigma_a
        self.sigma_x = sigma_x

        self.g = np.matrix(-9.8)

        self.x_p = []

        self.I = np.matrix(np.identity(2))

        # 状态变换矩阵
        self.F = np.matrix(np.array(
            [
                [1, self.delta_t],
                [0, 1]
            ]
        ))

        # 控制矩阵
        self.G = np.matrix(
            np.array(
                [[0.5 * self.delta_t ** 2], [self.delta_t]]
            )
        )

        # 过程噪声矩阵
        self.Q = np.matrix(np.array(
            [
                [self.delta_t**4 / 4, self.delta_t ** 3 / 2],
                [self.delta_t**3 / 2, self.delta_t**2]
            ]
        )) * self.sigma_a**2

        # 测量不确定性矩阵
        self.R = np.matrix(np.array(
            [
                self.sigma_x**2
            ]))

        # 观测矩阵
        self.H = np.matrix(np.array(
            [
                [1, 0]
            ]))

        self.x = np.matrix(np.zeros((2, 1)))

        self.P = np.matrix(np.zeros((2, 2)))
        self.P[0, 0] = self.P[1, 1] = 500

    def update(self, z):
        self.Kn = self.P.dot(self.H.transpose()).dot(
            (self.H.dot(self.P).dot(self.H.transpose()) + self.R).I)
        self.x = self.x + self.Kn.dot(z - self.H.dot(self.x))
        self.P = (self.I - self.Kn.dot(self.H)).dot(self.P)\
            .dot((self.I - self.Kn.dot(self.H)).transpose()) \
            + self.Kn.dot(self.R).dot(self.Kn.transpose())
        self.x_p.append(self.x[0, 0])

    def predict(self, a):
        self.x = self.F.dot(self.x) + self.G.dot(a + self.g)
        self.P = self.F.dot(self.P).dot(self.F.transpose()) + self.Q

# 火箭高度
x_measured = [-32.4, -11.1, 18, 22.9, 19.5, 28.5, 46.5, 68.9, 48.2, 56.1, 90.5, 104.9, 140.9, 148, 187.6, 209.2, 244.6, 276.4, 323.5,
              357.3, 357.4, 398.3, 446.7, 465.1, 529.4, 570.4, 636.8, 693.3, 707.3, 748.5]


a_measured = [39.72, 40.02, 39.97, 39.81, 39.75, 39.6, 39.77, 39.83, 39.73, 39.87, 39.81, 39.92, 39.78, 39.98, 39.76, 39.86, 39.61,
              39.86, 39.74, 39.87, 39.63, 39.67, 39.96, 39.8, 39.89, 39.85, 39.9, 39.81, 39.81, 39.68]


kf = KalmanFilter(0.25, 0.1, 20)
# 初始化后先预测n->n+1
z_0 = np.matrix(0)
a_0 = np.matrix(0)
kf.g = np.matrix(9.8)
kf.predict(a_0)
kf.g = np.matrix(-9.8)

for i in range(len(x_measured)):
    z_i = np.matrix(x_measured[i])
    a_i = np.matrix(a_measured[i])
    kf.update(z_i)
    kf.predict(a_i)

x = list(np.linspace(0, 0.25*30, 31))
print(x)
x.remove(x[-1])
y = x_measured
y_true = [0.5*30*i**2 for i in x]

plt.plot(x, y, 'bv-', label='Measured')
plt.plot(x, kf.x_p, 'ro-', label='Estimates')
plt.plot(x, y_true, 'g-', label='True')
plt.legend(loc='best')
plt.xlabel("t(s)")
plt.ylabel("altitude (m)")
plt.title("Kalman filter for vehicle estimates")
plt.show()
