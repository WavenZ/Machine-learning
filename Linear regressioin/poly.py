import matplotlib.pyplot as plt
import numpy as np

temp = np.loadtxt(open("data2.csv", "rb"), delimiter=",", skiprows=0)

# degree of polynomial
e = 3

# get data
m = temp.shape[0]
n = e + 1
x = np.ones((m, n))

for i in range(n):
    x[:, i] = np.power(temp[:, 0], i)

y = temp[:, 1].reshape(m, 1)

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
theta = np.ones(n).reshape(n, 1)

# init alpha
alpha = 0.1


# gradient descent
def update():
    global m, theta, x, y, J
    temp = np.dot(x, theta) - y  # h(x) - y
    J = (temp * temp).sum(axis=0) / (2 * m)  # cost_function
    dJ = (((temp * x).sum(axis=0)) / m).reshape(n, 1)  # dJ/dt
    theta = theta - alpha * dJ  # update

for i in range(10000000):
    update()


# line
linex = np.linspace(5, 55, 100)
liney = theta[0] + theta[1]*(linex-meanx[1])/rangex[1] + theta[2]*(linex*linex-meanx[2])/rangex[2] + theta[3]*(linex*linex*linex-meanx[3])/rangex[3]
# liney = theta[0] + theta[1]*(linex-meanx[1])/rangex[1]
line, = ax.plot(linex, liney, "-", alpha=0.5, color="gray")



# figure settings
plt.ylim(0, 10)
plt.xlim(0, 60)
plt.xlabel(r'$x$', size=16)
plt.ylabel(r'$y$', size=16)
plt.tick_params(labelsize=12)
plt.title("Polynomial regression", size=18)

# show
plt.show()
