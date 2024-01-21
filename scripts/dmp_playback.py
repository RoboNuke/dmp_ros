import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import rospkg
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import SetBool

from dmp_ros.dmp_utils import DMPUtils
from dmp_ros.discrete_dmp import DiscreteDMP
from dmp_ros.cs import CS
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

class DMPPlayback:
    def __init__(self):
        self.twistPub = rospy.Publisher('/bravo/servo_server/delta_twist_cmds', TwistStamped, queue_size=10)
        self.playbackService = rospy.Service("/start_playback", SetBool, self.startCB)
        self.playTraj = False

        self.pathPref = rospkg.RosPack().get_path('dmp_ros')
        self.trajPath = self.pathPref + rospy.get_param("trajectory_path", "/data/rest-looking_down-moveit.yaml")

        self.ee_link_name = rospy.get_param("ee_link", "ee_link")
        self.world_frame = rospy.get_param("world_frame", "bravo_base_link")

        self.nRBF = rospy.get_param("num_RBFs", 1000)
        self.by = rospy.get_param("Beta_Y", 25.0/4.0)
        self.ax = rospy.get_param("Alpha_X", 1.0)
        self.dt = rospy.get_param("dt",0.001)
        self.totalTime = rospy.get_param("total_time", 5.0)
        
        self.playing = False
        self.Ks = [1, -1, -1, 0.25, -0.8, -0.8]
        self.Ks = [2*x for x in self.Ks]

        self.tfBuffer = tf2_ros.Buffer()
        self.playing = False


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
        

    def getFramePose(self, frame):
        try:
            transform = self.tfBuffer.lookup_transform_core(self.world_frame, 
                                frame,
                                rospy.Time.now())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            #print("Failed to get TF")
            return None

        print("Got TF", rospy.Time.now())
        # get error term from goal
        pose = []
        pose.append(transform.transform.translation.x)
        pose.append(transform.transform.translation.y)
        pose.append(transform.transform.translation.z)
        q = []
        q.append(transform.transform.rotation.x)
        q.append(transform.transform.rotation.y)
        q.append(transform.transform.rotation.z)
        q.append(transform.transform.rotation.w)
        r,p,y = euler_from_quaternion(q)
        pose.append(r)
        pose.append(p)
        pose.append(y)
        return pose
    
    def calcError(self, A, B):
        e = 0
        for i in range(len(A)):
            e += (A[i] - B[i])**2 
        return sqrt(e)
    
    def pubDesiredTwist(self, robotPose):
        # calc control
        vels= []
        con = TwistStamped()
        for i in range(len(robotPose)):
            err = self.currentGoal - robotPose[i]
            vels.append(err * self.Ks[i])

        con.twist.linear.x = vels[0]
        con.twist.linear.y = vels[1]
        con.twist.linear.z = vels[2]
        con.twist.angular.x = vels[3]
        con.twist.angular.y = vels[4]
        con.twist.angular.z = vels[5]

        con.header.stamp = rospy.Time.now()
        self.lastTime = con.header.stamp
        con.header.frame_id = self.ee_link_name
        self.pubDesiredTwist(con)

    def startCB(self, req):
        if(req.data):
            self.loadDMPFromTraj()
            robotPose = self.getFramePose(self.ee_link_name)
            self.currentGoal = []
            for i in range(len(self.dmps)):
                self.dmps[i].reset(self.dmps[i].goal, robotPose[i])
                self.currentGoal.append(robotPose[i])

            self.lastTime = rospy.Time.now()
            self.pubDesiredTwist(robotPose)
            self.playing = True
            return True, "Starting Trajectory Playback"
        else:
            self.playing = False
            return False, "Stopping Trajectory Playback"
    
    def spin(self):
        while not rospy.is_shutdown():
            if self.playing:
                # get current state
                robotPose = self.getFramePose(self.ee_link_name)
                if robotPose == None:
                    continue

                # check if we need to update goal
                #coupled_error = self.calcError(robotPose, self.currentGoal)
                now = rospy.Time.now()
                if now - self.lastTime >= self.dt:
                    # step dmps forward (with error)
                    # note e is not currently used
                    for i in range(len(self.dmps)):
                        self.currentGoal[i] = self.dmps[i].step(self.totalTime)[0]
                    self.lastTime = now
                    
                self.pubDesiredTwist(robotPose)



if __name__=="__main__":
    rospy.init_node("dmp_playback_node")
    pb = DMPPlayback()
    rospy.spin()


