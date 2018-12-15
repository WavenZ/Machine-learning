import matplotlib.pyplot as plt
import numpy as np

temp = np.loadtxt(open("test.csv", "rb"), delimiter=",", skiprows=0)

# get data
m = temp.shape[0]
n = temp.shape[1] - 1 + 1
x = np.ones((m, n))
x[:, 1:] = temp[:, :-1]
y = temp[:, -1].reshape(m, 1)

# figure
fig, ax = plt.subplots(1, 1, figsize=(9, 6))

# plot data
ax.plot(x[:, 1], y, "x", alpha = 0.5, color="red")

# calc
meanx = x.mean(axis=0)
rangex = x.max(axis=0) - x.min(axis=0)

# feature scaling
x[:, 1:] = (x[:, 1:] - meanx[1:]) / rangex[1:]

# init theta
theta = np.ones(temp.shape[1]).reshape(n, 1)

# init alpha
alpha = 0.05


# gradient descent
def update():
    global m, theta, x, y, J
    temp = np.dot(x, theta) - y  # h(x) - y
    J = (temp * temp).sum(axis=0) / (2 * m)  # cost_function
    dJ = (((temp * x).sum(axis=0)) / m).reshape(n, 1)  # dJ/dt
    theta = theta - alpha * dJ  # update

for i in range(10000):
    update()

theta[0] = theta[0] - theta[1]*meanx[1]/rangex[1]
theta[1] = theta[1]/rangex[1]

print(theta)

# curve
linex = np.linspace(550, 4950, 10)
liney = theta[0] + theta[1]*linex
line, = ax.plot(linex, liney, "-", alpha=0.5, color="gray")


# figure settings
plt.ylim(-50, 1000)
plt.xlim(500, 5000)
plt.xlabel(r'$x$', size=16)
plt.ylabel(r'$y$', size=16)
plt.tick_params(labelsize=12)
plt.title("Linear regression", size=18)

# show
plt.show()
