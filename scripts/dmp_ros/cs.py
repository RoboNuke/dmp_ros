import numpy as np
import matplotlib.pyplot as plt

class CS():
    def __init__(self, ax, dt):
        self.run_time = 1.0
        self.dt = dt
        self.timesteps = int(self.run_time / self.dt)
        self.xPath = []
        self.ax = ax

    def resetState(self):
        self.x = 1.0

    def rollout(self):
        self.resetState()
        self.xPath = [0 for i in range(self.timesteps)]
        for t in range(self.timesteps):
            self.xPath[t] = self.x
            self.x  += -self.ax * self.x * self.dt
        return self.xPath
    
if __name__=="__main__":
    cs = CS(5.0)
    path = cs.rollout(100)
    t = np.linspace(0,1,len(path))
    plt.plot(t, path)
    plt.show()