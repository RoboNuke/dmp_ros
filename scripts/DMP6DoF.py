from dmp_ros.dmp_utils import DMPUtils
#from dmp_ros.oneD import DMP_1D
from dmp_ros.discrete_dmp import DiscreteDMP
import matplotlib.pyplot as plt
import numpy as np

import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool


fp = "/home/hunter/catkin_ws/src/dmp_ros/scripts/dmp_ros/rest-looking_down-moveit.yaml"
dt = 0.001
totT = 5
start = -1
trajs = []
ee_link_name = "ee_link"
world_frame = "bravo_base_link"
gains = [1, -1, -1, 0.25, -0.8, -0.8]
gains = [2*x for x in gains]
rollingOut = False
def toggleCB(msg):
    global rollingOut, start
    if(not msg.data and rollingOut):
        rollingOut = False
        start = -1
    rollingOut = msg.data
    print("Rolling Out:", rollingOut)

def getControl(robPose, trajs, gains, dt):
    global start
    # determine which waypoint to go towards
    
    # get current goal
    now = rospy.Time.now()
    
    idx = int((rospy.Time.now() - start).to_sec() / dt)
    #print(now, start, idx, dt)
    if idx > len(trajs[0]) - 1:
        start = -1
        rollingOut = False
        print("Completed Trajectory")
        return None
    vels= []
    con = TwistStamped()
    for i in range(len(robPose)):
        goal = trajs[i][idx]
        e = goal - robPose[i]
        vels.append(e * gains[i])
    con.twist.linear.x = vels[0]
    con.twist.linear.y = vels[1]
    con.twist.linear.z = vels[2]
    con.twist.angular.x = vels[3]
    con.twist.angular.y = vels[4]
    con.twist.angular.z = vels[5]
    print(idx, con.twist)
    con.header.stamp = rospy.Time.now()
    con.header.frame_id = ee_link_name
    
    return con

def getDMPs(fp):
    # load data
    ut = DMPUtils()
    traj = ut.loadTraj(fp)
    ts = traj[0]
    poses = traj[1]
    fts = traj[2]
    dmps = [DiscreteDMP(betaY=25.0/4.0, nRBF=1000, alphaX=1.0, dt = 0.001) for i in range(len(poses[0]))]


    # break in to x,y,ect
    poseTrajs = [[] for i in range(len(poses[0]))]
    for t in range(len(poses)):
        pose = poses[t]
        for i in range(len(pose)):
            poseTrajs[i].append(pose[i])

    # learn dmps for each
    for i, traj in enumerate(poseTrajs):
        dmps[i].start = traj[0]
        dmps[i].goal = traj[-1]
        #dmps[i].learnWeights(ts, traj)
        dmps[i].learnWeights(traj)
    
    return dmps, ts, poseTrajs


def saveDMPs(dmps, poseTrajs, ts):
    lab = ['x', 'y', 'z', 'r', 'p', 'ry']
    for i,traj in enumerate(poseTrajs):
        dmps[i].learnWeights(traj)
        its, izs = dmps[i].rollOut(traj[0],traj[-1])
        plt.figure(i)
        plt.plot(ts, traj)
        plt.plot(np.linspace(0, ts[-1], len(izs)), izs ,'r--')
        plt.xlabel("time(s)")
        plt.ylabel("Function")
        plt.title(f'DMP of {lab[i]}')
        plt.legend(['Original Trajectory', 'DMP'])
        plt.tight_layout()
        #plt.show()
        print(f"Saving {i} or {lab[i]}")
        plt.savefig(f'/home/hunter/catkin_ws/src/dmp_ros/scripts/dmp_ros/better_square/{lab[i]}_dmp.png')


if __name__=="__main__":
    dmps, ts, poseTrajs = getDMPs(fp)
    #saveDMPs(dmps, poseTrajs, ts)
    rospy.init_node('Op_Space_JS', anonymous=True)
    rate = rospy.Rate(25)
    tfBuffer = tf2_ros.Buffer()
    # Facilitates the receiving of transforms
    listener = tf2_ros.TransformListener(tfBuffer)
    pub = rospy.Publisher('/bravo/servo_server/delta_twist_cmds', TwistStamped, queue_size=10)
    sub = rospy.Subscriber('/startDMPRollout', Bool, toggleCB)
    while not rospy.is_shutdown():
        if rollingOut:
            # get rollout data
            if(start == -1):
                print("Starting Up")
                trajs = []
                for dmp in dmps:
                    trajs.append(dmp.rollout(dmp.goal, dmp.start)[1])
                dt = totT / len(trajs[0])
                start = rospy.Time.now()

            # get current end-effector frame
            try:
                transform = tfBuffer.lookup_transform_core(world_frame, 
                                    ee_link_name,
                                    rospy.Time.now())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                #print("Failed to get TF")
                continue
            print("Got TF", rospy.Time.now())
            # get error term from goal
            robPose = []
            robPose.append(transform.transform.translation.x)
            robPose.append(transform.transform.translation.y)
            robPose.append(transform.transform.translation.z)
            q = []
            q.append(transform.transform.rotation.x)
            q.append(transform.transform.rotation.y)
            q.append(transform.transform.rotation.z)
            q.append(transform.transform.rotation.w)
            r,p,y = euler_from_quaternion(q)
            robPose.append(r)
            robPose.append(p)
            robPose.append(y)

            # actual control
            con = getControl(robPose, trajs, gains, dt)
            print("Got control")
            if(con == None):
                continue
            pub.publish(con)
            print("Published control")
        rate.sleep()