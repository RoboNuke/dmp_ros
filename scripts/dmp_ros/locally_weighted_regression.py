import numpy as np
import matplotlib.pyplot as plt

class LWR():
    def __init__(self):
        pass


    def local_weighted_regression(self,x0, X, Y, tau):
        # add bias term
        x0 = np.r_[1, x0]
        X = np.c_[np.ones(len(X)), X]
        # fit model: normal equations with kernel
        xw = X.T * self.weights_calculate(x0, X, tau)
        theta = np.linalg.pinv(xw @ X) @ xw @ Y
        # "@" is used to
        # predict value
        return x0 @ theta
    
    # function to perform weight calculation
    def weights_calculate(self,x0, X, tau):
        return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * (tau **2) ))

if __name__ == "__main__":
    #define distribution
    n = 1000
    
    # generate dataset
    X = np.linspace(-3, 3, num=n)
    Y = np.abs(X ** 3 - 1)
    
    # jitter X
    X += np.random.normal(scale=.1, size=n) 
    #plt.plot(X,Y)
    #plt.show()
    # prediction
    lwr = LWR()
    domain = np.linspace(-3, 3, num=300)
    prediction = [lwr.local_weighted_regression(x0, X, Y, 0.1) for x0 in domain]
    plt.plot(X,Y)
    plt.plot(domain, prediction)
    plt.show()