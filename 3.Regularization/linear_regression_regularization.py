import matplotlib.pyplot as plt
import numpy as np

class polyRegression(object):

    cnt = 0
    def __init__(self, data, term, alpha, iter, lam):
        self.data = data
        self.term = term
        self.alpha  = alpha
        self.iter = iter
        self.lam = lam
    
    def dataProcess(self):
        self.m = data.shape[0]
        self.n = self.term + 1
        self.x = np.ones((self.m, self.n))
        for i in range(self.n):
            self.x[:, i] = np.power(self.data[:, 0], i)
        self.y = self.data[:, 1].reshape(self.m, 1)
        # calc
        self.meanx = self.x.mean(axis=0)
        self.rangex = self.x.max(axis=0) - self.x.min(axis=0)
        # feature scaling
        self.x[:, 1:] = (self.x[:, 1:] - self.meanx[1:]) / self.rangex[1:]

    def paramInit(self):
        self.theta = np.ones(self.n).reshape(self.n, 1)

    def iteration(self):
        temp = np.dot(self.x, self.theta) - self.y  # h(x) - y
        J = (1 / 2*self.m) * (temp * temp).sum(axis=0) # cost_function
        dJ = (1 / self.m) * (temp.T.dot(self.x)).T # dJ/dt
        dJ[1:] = dJ[1:] + (self.lam/self.m) * self.theta[1:]
        self.theta = self.theta - self.alpha * dJ  # update
        self.cnt = self.cnt + 1
        if self.cnt % 100 == 0:
            print(J)

    def regression(self):
        self.dataProcess()
        self.paramInit()
        for i in range(self.iter):
            self.iteration()
        return self.theta, self.meanx, self.rangex

def plotFig():
    # figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # plot data
    ax.plot(data[:, 0], data[:, 1], "x", alpha = 0.5, color="red")
    ax.plot(linex, liney, "-", alpha=0.5, color="gray")
    ax.plot(linex, liney1, "--", alpha=0.5)
    plt.xlim([0, 70])
    plt.ylim([0, 12])
    plt.xlabel(r'$x$', size=16)
    plt.ylabel(r'$y$', size=16)
    plt.tick_params(labelsize=12)
    plt.title("Polynomial regression", size=18) 
    plt.show()
    


if __name__ == "__main__":
    iter = 8
    data = np.loadtxt(open("data2.csv", "rb"), delimiter=",", skiprows=0)
    
    reg = polyRegression(data, iter, 0.2, 100000, 0)
    theta, meanx, rangex = reg.regression()
    linex = np.linspace(5, 65, 100)
    liney = np.ones_like(linex) * theta[0]
    for i in range(iter):
        liney = liney + theta[i+1] * (np.power(linex, i+1) - meanx[i+1]) / rangex[i+1]
    
    reg = polyRegression(data, iter, 0.2, 100000, 20)
    theta, meanx, rangex = reg.regression()
    liney1 = np.ones_like(linex) * theta[0]
    for i in range(iter):
        liney1 = liney1 + theta[i+1] * (np.power(linex, i+1) - meanx[i+1]) / rangex[i+1]
    
    plotFig()
