import rospy
from std_msgs.msg import Float64MultiArray
import rosbag
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

recording =  False
bag = None
joint_topic = "/bravo/joint_states"
count = 0

def jointStateCB(msg):
    global recording, bag, joint_topic
    if recording:
        print("Writting")
        bag.write(joint_topic, msg)

def recordSrvCB(req):
        global recording, count, bag
        if(req.data and not recording):
            bag = rosbag.Bag(f'joint_recording_{count}.bag','w')
            recording = True
            return True, "Starting collection joint state ROSBag"
        elif recording and not req.data:
            recording = False
            bag.close()
            count += 1
            return False, "Stopping Joint State Recording"
        elif recording and req.data:
            return True, "Already recording not starting a second bag"
        elif not recording and not req.data:
            return False, "Not recording, so nothing to stop"
        else:
            return False, "WTF did you do? Unsupported Behavior"

if __name__=="__main__":
    rospy.init_node("gravity_comp_node")
    zeroEffort = Float64MultiArray()
    zeroEffort.data = [0,0,0,0,0,0]
    sub = rospy.Subscriber(joint_topic, JointState, jointStateCB)
    pub = rospy.Publisher("/bravo/arm_gravity_compensation_controller/command", Float64MultiArray, queue_size=1)
    recordSRV = rospy.Service("/start_joint_recording", SetBool, recordSrvCB)
    while not rospy.is_shutdown():
        pub.publish(zeroEffort)
        rospy.Rate(20).sleep()

