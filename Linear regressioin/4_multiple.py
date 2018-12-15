import matplotlib.pyplot as plt
import numpy as np

temp = np.loadtxt(open("data1.csv", "rb"), delimiter=",", skiprows=0)

# get data
m = temp.shape[0]
n = temp.shape[1] - 1 + 1
x = np.ones((m, n))
x[:, 1:] = temp[:, :-1]
y = temp[:, -1].reshape(m, 1)

# calc
meanx = x.mean(axis=0)
rangex = x.max(axis=0) - x.min(axis=0)

# feature scaling
x[:, 1:] = (x[:, 1:] - meanx[1:]) / rangex[1:]

# init theta
theta = np.ones(temp.shape[1]).reshape(n, 1)

# init alpha
alpha = 0.1


# gradient descent
def update():
    global m, theta, x, y, J
    temp = np.dot(x, theta) - y  # h(x) - y
    # J = (temp * temp).sum(axis=0) / (2 * m)  # cost_function
    J = np.dot(temp.T, temp)/(2 * m)
    dJ = (((temp * x).sum(axis=0)) / m).reshape(n, 1)  # dJ/dt
    theta = theta - alpha * dJ  # update

for i in range(10000):
    update()

for i in range(n-1):
    theta[0] = theta[0] - theta[i+1]*meanx[i+1]/rangex[i+1]

for i in range(n-1):     
    theta[i+1] = theta[i+1]/rangex[i+1]

print(theta)