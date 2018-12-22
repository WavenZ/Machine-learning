import matplotlib.pyplot as plt
import numpy as np 

class logisRegression(object):
    
    def __init__(self, theta, alpha):
        self.theta = theta
        self.alpha = alpha
        self.cos = 0

    def sigmoid(self, x):
        z = 1/(1+np.exp(-x))
        return z

    def cost(self, trainX, trainY):
        self.cos = (1/m)*(self.sigmoid(trainX.dot(self.theta))-trainY).T.dot(trainX).T

    def update(self, trainX, trainY):
        self.cost(trainX, trainY)
        self.theta = self.theta - self.alpha * self.cos


def plotFig(lr):
    plt.plot(data[:15, 0], data[:15, 1], 'x')
    plt.plot(data[15:, 0], data[15:, 1], 'x')
    plt.xlim([-3, 5])
    plt.ylim([-1, 5])
    a = np.linspace(-3, 5, 100)
    b = np.linspace(-1, 5, 100)
    x = np.meshgrid(a, b)
    z = 1/(1+np.exp(-(x[0]*lr.theta[1]+x[1]*lr.theta[2]+lr.theta[0])))
    plt.contour(x[0], x[1], z, [0.5], width=0.5, alpha=0.4)
    plt.show()



if __name__ == '__main__':
    data = np.loadtxt(open("1.txt", "rb"))
    m = data.shape[0]
    n = data.shape[1] + 1 - 1
    trainX = np.ones(data.shape)
    trainX[:, 1:] = data[:, :-1]
    trainY = data[:, -1].reshape(data.shape[0], 1)
    theta = np.zeros((n, 1))
    LR = logisRegression(theta, 0.03)
    for i in range(30000):
        LR.update(trainX, trainY)
    print(LR.theta)
    plotFig(LR)