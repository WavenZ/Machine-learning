import matplotlib.pyplot as plt
import numpy as np

temp = np.loadtxt(open("data3.csv", "rb"), delimiter=",", skiprows=0)

# get data
m = temp.shape[0]
n = temp.shape[1] - 1 + 1
x = np.ones((m, n))
x[:, 1:] = temp[:, :-1]
y = temp[:, -1].reshape(m, 1)

# calc
temp = np.linalg.inv(np.dot(x.T, x))
temp = np.dot(temp, x.T)
temp = np.dot(temp, y)

theta = temp
print(theta)
