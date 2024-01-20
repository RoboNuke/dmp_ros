from math import exp
import numpy as np

class RBF():
    def __init__(self, c, h):
        self.c = c
        self.h = h

    def eval(self, x):
        dx = x - self.c
        return exp(-self.h * (dx ** 2))

    def dEval(self, x):
        dx = self.c - x
        return 2 * self.h * dx * self.eval(x)

    def ddEval(self, x):
        y = 2*self.c**2*self.h
        y -= 4*self.c * self.h * x
        y += 2*self.h*x**2 - 1
        y *= self.eval(x)
        y *= 2* self.h 
        return y
    
    def theeMat(self, t):
        mat = np.zeros(len(t))
        for i in range(len(t)):
            mat[i] = self.eval(t[i])

        return mat