import numpy as np
import matplotlib.pyplot as plt


class logisRegression(object):
    def __init__(self, data, alpha, lam, degree, iter):
        self.data = data
        self.alpha = alpha  # learning rate
        self.lam = lam  # regularizatoin rate
        self.degree = degree  # degree of fitting curve
        self.iter = iter  # iterations

    def mapFeature(self, data):
        self.trainX = np.ones((data.shape[0], 1))
        for i in range(self.degree):  # 0..5 + 1
            for j in range(i + 2):  # 0 0..1 ... 0..5
                temp = np.power(data[:, 0], i + 1 - j) * np.power(
                    data[:, 1], j)
                self.trainX = np.c_[self.trainX, temp]
        self.trainY = data[:, 2].reshape(data.shape[0], 1)
        self.theta = np.ones((self.trainX.shape[1], 1))
        return self.trainX, self.trainY

    def sigmoid(self, x):
        z = 1 / (1 + np.exp(-x))
        return z

    def update(self):
        # Partial derivative of the cost function
        self.cost = (1/self.data.shape[0])*(self.sigmoid(self.trainX.dot(self.theta))\
                    -self.trainY).T.dot(self.trainX).T
        # Regularization
        self.cost[1:] = self.cost[1:] + (
            self.lam / self.data.shape[0]) * self.theta[1:]
        # Update theta
        self.theta = self.theta - self.alpha * self.cost

    def regression(self):
        self.mapFeature(self.data)
        for i in range(self.iter):
            self.update()
        return self.theta


def plotFig(data, X, Y, Z):
    a = data[np.where(data[:, 2] == 1)]
    b = data[np.where(data[:, 2] == 0)]
    plt.subplots(1, 1, figsize=(8, 5))
    plt.plot(a[:, 0], a[:, 1], 'x')  # group A
    plt.plot(b[:, 0], b[:, 1], 'x')  # group B
    plt.contour(X, Y, Z, [0.5], Width=0.5, alpha=0.4)  # decision boundary
    plt.xlabel(r'$x_1$', size=16)
    plt.ylabel(r'$x_2$', size=16)
    plt.tick_params(labelsize=12)
    plt.title("Logistic regression", size=18)
    plt.show()


def generateTestData():
    a = np.linspace(-1, 1.2, 100)
    b = np.linspace(-1, 1.2, 100)
    X = np.meshgrid(a, b)
    temp = np.ones((X[0].size, 1))
    testData = np.c_[X[0].reshape(X[0].size, 1), X[1].reshape(X[0].
                                                              size, 1), temp]
    return testData, X


if __name__ == "__main__":

    # data
    trainData = np.loadtxt(open("ex2data2.txt", "rb"), delimiter=",")

    # train
    f = logisRegression(trainData, 0.5, 0.2, 8,
                        300000)  # data, alpha, lam, degree, iter
    theta = f.regression()

    # test
    [testData, X] = generateTestData()
    [testX, testY] = f.mapFeature(testData)
    testZ = f.sigmoid(testX.dot(theta)).reshape(X[0].shape)

    # plot
    plotFig(trainData, X[0], X[1], testZ)
