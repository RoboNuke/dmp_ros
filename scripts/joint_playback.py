import rospy
from std_msgs.msg import Float64MultiArray
import rosbag
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from controller_manager_msgs.srv import SwitchController
from dmp_ros.srv import jointPlaybackReq
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

recording =  False
bag = None
count = 0

joint_topic = "/bravo/joint_states"
gravity_comp_cmd_topic = "/bravo/arm_gravity_compensation_controller/command"

controller_switch_name = "/bravo/controller_manager/switch_controller"
joint_control_controller = "arm_position_controller"
gravity_comp_controller = "arm_gravity_compensation_controller"

global_bag_path = "/home/hunter/catkin_ws/src/dmp_ros/data/"
#global_bag_path = ""
# joint trajectory action server stuff
joint_client_name = "/bravo/arm_position_controller/follow_joint_trajectory"

def jointStateCB(msg):
    global recording, bag, joint_topic
    if recording:
        bag.write(joint_topic, msg)

def recordSrvCB(req):
        global recording, count, bag, switch_controller
        if(req.data and not recording):
            # turn on gravity comp controller
            ret = switch_controller([gravity_comp_controller], [joint_control_controller], 2, True, 5.0)
            bag = rosbag.Bag(global_bag_path + f'joint_recording_{count}.bag','w')
            recording = True
            print(f"Starting collection joint state ROSBag {count}")
            return True, f"Starting collection joint state ROSBag {count}"
        elif recording and not req.data:
            #  turn off gravity comp controller
            ret = switch_controller([joint_control_controller], [gravity_comp_controller], 2, True, 5.0)
            
            recording = False
            bag.close()
            count += 1
            print("Stopping Joint State Recording")
            return False, "Stopping Joint State Recording"
        elif recording and req.data:
            return True, "Already recording not starting a second bag"
        elif not recording and not req.data:
            return False, "Not recording, so nothing to stop"
        else:
            return False, "WTF did you do? Unsupported Behavior"

def jointPlaybackCB(req):
    global recording
    if recording:
        print("Currently recording trajectory, complete that first!")
        return False, "Currently recording trajectory, complete that first!"
    
    # load bag file
    bag = rosbag.Bag(global_bag_path + req.bagFilePath)

    # extract joint trajectory
    tS = req.timeScaling
    first = True
    JT = JointTrajectory()
    lastTime = None
    for topic, msg, t in bag.read_messages(topics=[joint_topic]):
        if first:
            JT.joint_names = msg.name[1:]
            JT.header.frame_id = msg.header.frame_id
            t0 = msg.header.stamp
            lastTime = t0
            first = False
        
        # create joint point
        JP = JointTrajectoryPoint()
        JP.positions = msg.position[1:]
        
        # scale velocities 
        vels = msg.velocity[1:]
        for vel in vels:
            vel/=tS
            # set accel to zero
            #JP.accelerations.append(0.0)
        JP.velocities = vels

        # set time
        orgTime = msg.header.stamp
        time = orgTime - t0
        #time.secs *= tS
        #time.nsecs *= tS
        JP.time_from_start = time
        lastTime = orgTime

        # add to traj
        JT.points.append(JP)
        
 
    # run joint trajectory
    JT.header.stamp = rospy.Time.now()
    JT.header.stamp.secs += 2

    goal = FollowJointTrajectoryGoal()
    goal.trajectory = JT

    jointClient.send_goal(goal)

    jointClient.wait_for_result()

    res = jointClient.get_result()
    if(res.error_code == 0):
        return True, "Trajectory completed"
    else:
        return False, f"Trajectory failed to playback with error code:{res.error_code}"


if __name__=="__main__":
    rospy.init_node("gravity_comp_node")
    zeroEffort = Float64MultiArray()
    zeroEffort.data = [0,0,0,0,0,0]
    sub = rospy.Subscriber(joint_topic, JointState, jointStateCB)
    pub = rospy.Publisher(gravity_comp_cmd_topic, Float64MultiArray, queue_size=1)

    recordSRV = rospy.Service("/start_joint_recording", SetBool, recordSrvCB)
    jointPlaybackSRV = rospy.Service("/playback_joint_bag", jointPlaybackReq, jointPlaybackCB)

    jointClient = actionlib.SimpleActionClient(joint_client_name, FollowJointTrajectoryAction)
    jointClient.wait_for_server()

    # get controller manager service
    rospy.wait_for_service(controller_switch_name)

    switch_controller = rospy.ServiceProxy(controller_switch_name, SwitchController)
    while not rospy.is_shutdown():
        if recording:
            pub.publish(zeroEffort)
        rospy.Rate(20).sleep()