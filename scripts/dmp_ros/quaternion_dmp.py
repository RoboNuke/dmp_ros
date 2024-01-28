from math import exp
import matplotlib.pyplot as plt
import numpy as np
from dmp_ros.rbf import RBF
from dmp_ros.cs import CS


class QuaternionDMP():
    def __init__(self, nRBF=100, betaY=1, dt= 0.001, cs=CS(1.0, 0.001), RBFData=None, ws=None):