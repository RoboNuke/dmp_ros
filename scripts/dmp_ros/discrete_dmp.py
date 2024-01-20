from math import exp
import matplotlib.pyplot as plt
import numpy as np
from dmp_ros.rbf import RBF
from dmp_ros.cs import CS


class DiscreteDMP():
    def __init__(self, betaY=1, nRBF=100, alphaX=-1.0, dt = 0.001, data = None):
        if data == None:
            self.nRBF = nRBF
            self.ay = 4 * betaY
            self.by = betaY
            self.ax = alphaX
            self.dt = dt
            self.w = [1.0 for x in range(self.nRBF)]
            self.cs = CS(self.ax, self.dt)
            self.centerRBFs()
        else:
            self.by = data['by']
            self.ay = 4 * self.by
            self.nRBF = len(data['weights'])
            self.w = data['weights']
            self.RBFs = []
            for i in range(self.nRBF):
                c = data['centers'][i]
                h = data['widths'][i]
                self.RBFs.append(RBF(c,h))

            self.dt = data['dt']
            self.ax = data['ax']
            self.cs = CS(self.ax, self.dt)


    def centerRBFs(self):
        self.RBFs = []
        #des_c = np.linspace(min(self.cs.xPath), max(self.cs.xPath), self.nRBF)
        des_c = np.linspace(0, self.cs.run_time, self.nRBF)
        for i in range(self.nRBF):
            c = np.exp(-self.cs.ax * des_c[i])
            #c = des_c[i]
            #h = (self.nRBF) / (c ** 2)
            h = self.nRBF ** 1.5 / c / self.cs.ax
            self.RBFs.append(RBF(c,h))

    def learnWeights(self,y): #, ydot, ydotdot, tt):
        g = y[-1]
        x = np.array(self.cs.rollout())
        #dt = t / t[-1]
        path = np.zeros(len(x))
        ts = np.zeros(len(x))
        t=np.linspace(0, self.cs.run_time, len(y))
        import scipy.interpolate
        path_gen = scipy.interpolate.interp1d(t, y)
        for i in range(len(x)):
            path[i] = path_gen(i * self.dt)
            ts[i] = i * self.dt

        
        # estimate the gradients
        #ddt = self.dt * t[-1]
        dy = np.gradient(path)/(self.dt)
        ddy = np.gradient(dy)/(self.dt)
        """
        print(min(tt), max(tt), min(ts), max(ts))
        plt.plot(ts, path, '-r')
        plt.plot(tt/tt[-1], y)
        plt.show()
        plt.plot(ts ,dy, '-r')
        plt.plot(tt/tt[-1], ydot)
        plt.show()
        plt.plot(ts,ddy, '-r')
        plt.plot(tt/tt[-1], ydotdot)
        plt.show()
        """
        y = path
        rbMats = [self.RBFs[i].theeMat(x) for i in range(self.nRBF)]

        fd = ddy - self.ay * (self.by * (g - y) - dy)
        
        for i in range(len(self.w)):
            bot = np.sum( x ** 2 * rbMats[i])
            #bot = np.sum(x * rbMats[i])
            top = np.sum( x * rbMats[i] * fd)
            self.w[i] = top / bot
        #print(g - y[0])
        if abs(g - y[0]) > 0.0001:
            for i in range(self.nRBF):
                self.w[i]/= (g-y[0])

    def calcF(self, x):
        top = 0
        bot = 0
        for i in range(len(self.w)):
            thee = self.RBFs[i].eval(x)
            top += thee * self.w[i] 
            bot += thee
        return top/bot
    
    def rollout(self, g, y0, dy0=0, ddy0=0, tau=1, scale=1):
        y = y0
        ydot = dy0
        ydotdot = ddy0
        t = 0.0
        x = 1.0
        z = [y0]
        dz = [ydot]
        ddz = [ydotdot]
        #print(self.dt)
        ts = [0.0]
        #print(f"Total Time:{self.cs.run_time * tau}")
        while x > self.cs.xPath[-1]:
            xdot = -self.ax * x 
            x += xdot * self.dt / tau
            t+=self.dt

            ydotdot = self.ay * (self.by * (g -  y) - ydot) +  x*(g-y0)*self.calcF(x)
            ydot += ydotdot * self.dt / tau
            y += ydot * self.dt / tau

            z.append(y)
            dz.append(ydot)
            ddz.append(ydotdot)
            ts.append(t)
        z = np.array(z)
        dz = np.array(dz)/tau
        #dz[0]*=tau
        ddz = np.array(ddz)/(tau**2)
        #ddz[0]*=tau**2
        return(ts, z, dz, ddz)


def plotSub(ax, t, ts, org, cpy, tit="DMP", ylab="Function"):
    ax.plot(t, org)
    ax.plot(ts, cpy,'r--')
    #ax.plot(np.linspace(0, t[-1], len(cpy)), cpy,'r--')
    ax.set(ylabel=ylab)
    ax.set_title(tit,  fontsize=20)

if __name__=="__main__":
    dt = 0.001
    tmax = 5

    dmp = DiscreteDMP(25.0/4.0, 1000, 1.0, dt=dt)

    t = np.arange(0, tmax, dt)
    of = 0.5
    y = np.sin(of* 10*t) + np.cos(of * 3 *t)
    dy = of * 10*np.cos(of* 10*t) - of*3*np.sin(of*3*t)
    ddy = -100* of**2 * np.sin(of*10*t) - 9 * of**2 * np.cos(of*3*t)

    dmp.learnWeights(y) #,dy,ddy,t)

    tau = 1
    scale = 1
    g = y[-1] * scale

    ts, z, dz, ddz = dmp.rollout(g, y[0], dy[0]*tmax, ddy[0]*tmax**2, tau, scale)
    
    fig, axs = plt.subplots(3)
    fig.set_figwidth(800/96)
    fig.set_figheight(1000/96)
    fig.tight_layout(pad=5.0)

    plotSub(axs[0], t*tau/tmax, ts, y, z,"Position DMP", "Position")
    plotSub(axs[1], t*tau/tmax, ts, dy*(tmax/tau), dz, "Velocity DMP", "Velocity")
    plotSub(axs[2], t*tau/tmax, ts, ddy*( (tmax/tau)**2), ddz, "Accel DMP", "Acceleration")
    plt.xlabel("time (s)")

    plt.legend(['Original Function', 'Learned Trajectory'])
    plt.show()