import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio


class MultiClassification(object):
    def __init__(self, data, label, classes, alpha, lam, iter):
        self.m = label.shape[0]
        self.n = data.shape[1] + 1
        self.data = np.c_[np.ones((self.m, 1)), data]
        self.label = label
        self.alpha = alpha
        self.lam = lam
        self.iter = iter
        self.classes = classes
        self.theta = np.zeros((classes, self.n))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def costFunc(self, i):
        theta = self.theta[i, :].reshape(self.theta.shape[1], 1)
        label = (self.label == i + 1).astype('int16')
        data = self.data

        h = self.sigmoid(data.dot(theta))
        J = (-1/self.m) * (label.T.dot(np.log(h))\
                            +(1-label).T.dot(np.log(1-h)))\
                     + (self.lam/2/self.m) * theta.T.dot(theta)   # regularization
        grad = (1 / self.m) * data.T.dot(h - label)
        grad[1:] = grad[1:] + (self.lam / self.m * theta[1:])  # regularization
        return J, grad

    def regression(self):
        for i in range(self.classes):
            for j in range(self.iter):
                [J, grad] = self.costFunc(i)
                self.theta[i, :] = self.theta[i, :] - self.alpha * grad.T
            print(J)
        return self.theta

    def predict(self):
        pred = self.sigmoid(self.data.dot(self.theta.T)).argmax(1) + 1
        accuracy = np.mean(
            (self.label.flatten() == pred).astype('int16')) * 100
        print('accuracy:', accuracy)
        pass


def dataRead():
    data = scio.loadmat('data.mat')
    X = data['X']
    y = data['y']
    print(X.shape, y.shape)
    return X, y


def dataVisualize(data):  # combine them into a 1000*2000 image
    temp = data[0, :].reshape(20, 20)
    print(data.shape)
    for i in range(4999):
        temp = np.c_[temp, data[i + 1, :].reshape(20, 20)]
    print(temp.shape)
    temp1 = temp[:, :2000]
    for i in range(49):
        temp1 = np.r_[temp1, temp[:, 2000 * (i + 1):2000 * (i + 2)]]
    print(temp1.shape)
    plt.imshow(temp1)
    plt.show()


if __name__ == "__main__":
    [data, label] = dataRead()
    # dataiVsualize(data)
    f = MultiClassification(data, label, 10, 1, 0.1, 10000)
    f.regression()
    f.predict()
