import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import rospkg
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

from dmp_ros.dmp_utils import DMPUtils
from dmp_ros.discrete_dmp import DiscreteDMP
from dmp_ros.cs import CS
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt

class DMPPlayback:
    def __init__(self):
        self.pathPref = rospkg.RosPack().get_path('dmp_ros')
        self.trajPath = self.pathPref + rospy.get_param("trajectory_path", "/data/rest-looking_down-moveit.yaml")

        self.ee_link_name = rospy.get_param("ee_link", "ee_link")
        self.world_frame = rospy.get_param("world_frame", "bravo_base_link")

        self.nRBF = rospy.get_param("num_RBFs", 1000)
        self.by = rospy.get_param("Beta_Y", 25.0/4.0)
        self.ax = rospy.get_param("Alpha_X", 1.0)
        self.dt = rospy.get_param("dt",0.001)
        self.totalTime = rospy.get_param("total_time", 1)
        
        self.Kps = [10, 10, 10, -4.5, -15, 10.0]
        self.Kds = [1, 1, 1, -0.45, -1.5, 1.0]
        self.Kds = [0 for i in range(6)]
        #self.Kds = self.Kps
        self.playing = False
        self.tuning = False
        self.tuneIdx = 5
        self.poseGoals = [0.0 for i in range(6)]
        self.poseGoals[self.tuneIdx] = -2.15
        self.velGoals = [0.0 for i in range(6)]
    
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.twistPub = rospy.Publisher('/bravo/servo_server/delta_twist_cmds', TwistStamped, queue_size=10)
        self.eeStateSub = rospy.Subscriber('/bravo/ee_state', JointState, self.eeStateCB)
        self.playbackService = rospy.Service("/start_playback", SetBool, self.startCB)


    def eeStateCB(self, msg):
        self.pose = msg.position
        self.vels = msg.velocity
        self.updated = True

        if(self.playing):
            now = rospy.Time.now()
            if not self.tuning:

                if (now - self.lastTime).to_sec() > self.dt:
                    for i in range(6):
                        y, dy, ddy = self.dmps[i].step(self.totalTime)
                        self.poseGoals[i] = y
                        self.velGoals[i] = dy
            else:
                if (now - self.startTime).to_sec() > 5.0:
                    self.playing = False
                    print(len(self.ts))
                    plt.plot(self.ts, self.locs)
                    plt.plot(self.ts, self.goals)
                    #plt.legend("Plant", "Goal")
                    plt.show()


            us, errs, dErrs = self.pubDesiredTwist()
            out = ""
            #out += f"{self.poseGoals[self.tuneIdx]:1.3f}, {self.pose[self.tuneIdx]:1.3f}, {errs[self.tuneIdx]:1.3f}"
            #self.ts.append((now - self.startTime).to_sec())
            #self.goals.append(self.poseGoals[self.tuneIdx])
            #self.locs.append(self.pose[self.tuneIdx])
            
            for i, err in enumerate(errs):
                if i < 3:
                    out += f"{err:1.3f}\t"
                else:
                    out +=f"{err*180/pi:1.3f}\t"
            
            print(out)

    def loadDMPFromTraj(self):
        # load data
        ut = DMPUtils()
        traj = ut.loadTraj(self.trajPath)
        poses = traj[1]
        nTraj = len(poses[0])
        #self.cs = CS(self.ax, self.dt)
        self.dmps = [DiscreteDMP(nRBF=self.nRBF, betaY=self.by, dt = self.dt, cs = CS(self.ax, self.dt)) for i in range(nTraj)]

        # break in to x,y,ect
        self.poseTrajs = [[] for i in range(nTraj)]
        for t in range(len(poses)):
            pose = poses[t]
            for i in range(len(pose)):
                self.poseTrajs[i].append(pose[i])

        # learn dmps for each
        for i, traj in enumerate(self.poseTrajs):
            self.dmps[i].learnWeights(traj)
    
    def calcError(self, A, B):
        e = 0
        for i in range(len(A)):
            e += (A[i] - B[i])**2 
        return sqrt(e)
    
    def angleDiff(self, a, b):
        # at b going to a
        diff = a - b
        if abs(diff) > pi:
            if diff < 0:
                diff += 2 * pi
            else:
                diff -= 2 * pi
        return diff
    
    def pubDesiredTwist(self):

        us = [0.0 for i in range(6)]
        errs = [0.0 for i in range(6)]
        dErrs = [0.0 for i in range(6)]
        for i in range(6):
            if i < 3:
                err = self.poseGoals[i] - self.pose[i]
                derr = self.velGoals[i] - self.vels[i]
            else:
                err = self.angleDiff(self.poseGoals[i], self.pose[i])
                derr = self.velGoals[i] - self.vels[i]

            us[i] = self.Kps[i] * err + self.Kds[i] * derr
            errs[i] = err
            dErrs[i] = derr

        con = TwistStamped()

        con.twist.linear.x = us[0]
        con.twist.linear.y = us[1]
        con.twist.linear.z = us[2]
        con.twist.angular.x = us[3]
        con.twist.angular.y = us[4]
        con.twist.angular.z = us[5]

        con.header.stamp = rospy.Time.now()
        self.lastTime = con.header.stamp
        #con.header.frame_id = self.ee_link_name
        con.header.frame_id = self.world_frame
        self.twistPub.publish(con)

        return us, errs, dErrs


    def startCB(self, req):
        if(req.data):
            self.loadDMPFromTraj()
            print("DMP Loaded")

            self.updated = False
            while self.updated == False:
                rospy.Rate(1.0/self.dt).sleep()

            print("State Recieved")
            for i in range(len(self.dmps)):
                self.dmps[i].reset(self.dmps[i].goal, self.pose[i])
                self.poseGoals[i] = self.dmps[i].y
                self.velGoals[i] = self.dmps[i].dy

            print("Goals Set")
            self.lastTime = rospy.Time.now()
            self.playing = True

            return True, "Starting Trajectory Playback"
        else:
            self.lastTime = rospy.Time.now()
            self.startTime = self.lastTime
            self.ts = []
            self.goals = []
            self.locs = []
            self.playing = True
            self.tuning = True
            return False, "Stopping Trajectory Playback"



if __name__=="__main__":
    rospy.init_node("dmp_playback_node")
    pb = DMPPlayback()
    rospy.spin()


