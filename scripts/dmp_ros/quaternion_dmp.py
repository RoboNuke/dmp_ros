from math import exp
import matplotlib.pyplot as plt
import numpy as np
from dmp_ros.rbf import RBF
from dmp_ros.cs import CS


class QuaternionDMP():
    def __init__(self, nRBF=100, betaY=1, dt= 0.001, cs=CS(1.0, 0.001), RBFData=None, ws=None):
        pass

    def centerRBFs(self):
        pass

    def learnWeights(self, q):
        pass
    
    def step(self, tau=1.0, error=0.0, external_force=None):
        pass

    def reset(self, goal, q, w = [0.0]*3, dw = [0.0]*3):
        pass

    def rollout(self, g, q0, w0=[0.0]*3, dw0=[0.0]*3, tau=1, scale=1):
        pass


