import numpy as np
import time

# Pinocchio
import pinocchio as pin
from pinocchio import RobotWrapper
from pinocchio.utils import *


class jointTraj:
    def __init__(self, jointNum):
        self.n_joint = jointNum

        # Timer
        self.ctime = None
        self.stime = None
        self.duration = None

        # init 
        self.init : np.array

        # Target
        self.target : np.array

    def setCurrentTime(self, ctime):
        self.ctime = ctime

    def setDuration(self, duration):
        self.duration = duration

    def setStartTime(self, stime):
        self.stime = stime

    def setTargetSample(self, target : np.array):
        self.target = np.copy(target)

    def setInitSample(self, init : np.array):
        self.init = np.copy(init)

    def computeNext(self):
        qCubic = np.zeros((self.init.shape[0]))
        if(self.ctime < self.stime):
            return self.init
        elif(self.ctime > self.stime + self.duration):
            return self.target
        else:
            for i in range(self.init.shape[0]):
                a0 = self.init[i]
                a1 = 0.0
                a2 = 3/(self.duration)**2 * (self.target[i] - self.init[i])
                a3 = -1.0 * 2.0/(self.duration)**3 * (self.target[i]- self.init[i]) 
                qCubic[i] = a0 + a1 * (self.ctime - self.stime) + a2 * (self.ctime - self.stime)**2 + a3 * (self.ctime - self.stime)**3
        return qCubic


class SE3Traj:
    def __init__(self):
        # Timer
        self.ctime = None
        self.stime = None
        self.duration = None

        # Init
        self.init : pin.SE3
        self.init_6d = np.zeros((6))

        # Target
        self.target : pin.SE3
        self.target_6d = np.zeros((6))

    def setCurrentTime(self, ctime):
        self.ctime = ctime

    def setDuration(self, duration):
        self.duration = duration

    def setStartTime(self, stime):
        self.stime = stime

    def setTargetSample(self, target : pin.SE3):
        self.target = target
        self.target_6d[:3] = self.target.translation
        self.target_6d[3:] = pin.log3(self.target.rotation)

        


    def setInitSample(self, init : pin.SE3):
        self.init = init
        self.init_6d[:3] = self.init.translation
        self.init_6d[3:] = pin.log3(self.init.rotation)

    def computeNext(self):
        se3Cubic_6d = np.zeros((6)) # xyz : 3 , rpy : 3


        if(self.ctime < self.stime):
            return self.init
        elif(self.ctime > self.stime + self.duration):
            return self.target
        else:
            for i in range(6):
                a0 = self.init_6d[i]
                a1 = 0.0
                a2 = 3.0/(self.duration)**2 * (self.target_6d[i] - self.init_6d[i])
                a3 = -1.0 * 2.0/(self.duration)**3 * (self.target_6d[i]- self.init_6d[i]) 
                se3Cubic_6d[i] = a0 + a1 * (self.ctime - self.stime) + a2 * (self.ctime - self.stime)**2 + a3 * (self.ctime - self.stime)**3
                
        se3Cubic = pin.SE3()
        se3Cubic.translation = se3Cubic_6d[:3]
        se3Cubic.rotation = pin.exp3(se3Cubic_6d[3:])

        
        return se3Cubic
