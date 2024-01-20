import yaml
import rosbag
import numpy as np
import tf2_py
import rospy
import sys
from dmp_ros.discrete_dmp import DiscreteDMP
from dmp_ros.cs import CS
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class DMPUtils():
    def __init__(self):
        self.ee_link_name = "ee_link"
        self.world_frame = "bravo_base_link"
        self.ft_topic = "/bravo/raw_force_torque"
        self.useQuaternion = False

    def extractTFBuffer(self, bag):
        tf_buffer = tf2_py.BufferCore(rospy.Duration(1000000))

        for topic, msg, t in bag.read_messages(topics=['/tf', '/tf_static']):
            for msg_tf in msg.transforms:
                if topic == '/tf_static':
                    tf_buffer.set_transform_static(msg_tf, "default_authority")
                else:
                    tf_buffer.set_transform(msg_tf, "default_authority")
        return tf_buffer
    
    def floatDrop(self, ft):
        return ft if ft > 0.0001 else 0
    

    def parseRosbag(self, bagpath):
        #"Parses a rosbag and creates a trajectory"
        ts = []
        poses = []
        fts = []

        bag = rosbag.Bag(bagpath)
        # get tf buffer
        tf_buffer = self.extractTFBuffer(bag)
        # at each ft_sensor time get location
        count = 0 
        for topic, msg, t in bag.read_messages(topics=[self.ft_topic]):
            if count < 10:
                count += 1
                continue
            ts.append(msg.header.stamp.secs + msg.header.stamp.nsecs/1000000000.0)
            ft = []
            ft.append(self.floatDrop(msg.wrench.force.x))
            ft.append(self.floatDrop(msg.wrench.force.y))
            ft.append(self.floatDrop(msg.wrench.force.z))
            ft.append(self.floatDrop(msg.wrench.torque.x))
            ft.append(self.floatDrop(msg.wrench.torque.y))
            ft.append(self.floatDrop(msg.wrench.torque.z))
            fts.append(ft)
            transform = tf_buffer.lookup_transform_core(self.world_frame, 
                            self.ee_link_name,
                            msg.header.stamp)#,
            #                rospy.Duration(1))
            pose = []
            pose.append(transform.transform.translation.x)
            pose.append(transform.transform.translation.y)
            pose.append(transform.transform.translation.z)
            if(self.useQuaternion):
                pose.append(transform.transform.rotation.x)
                pose.append(transform.transform.rotation.y)
                pose.append(transform.transform.rotation.z)
                pose.append(transform.transform.rotation.w)
            else:
                q = []
                q.append(transform.transform.rotation.x)
                q.append(transform.transform.rotation.y)
                q.append(transform.transform.rotation.z)
                q.append(transform.transform.rotation.w)
                r,p,y = euler_from_quaternion(q)
                pose.append(r)
                pose.append(p)
                pose.append(y)
                
            poses.append(pose)
        t0 = ts[0]
        ts = [x-t0 for x in ts] # rescale 
        print(f"Have {ts[-1]} seconds of data over {len(ts)} data points")
        
        return( (ts, poses, fts) )

    def saveDMP(self, fp, dmp):
        data = {}
        ws = []
        cs = []
        hs = []
        for w in dmp.w:
            ws.append(w)

        for rbf in dmp.RBFs:
            c = rbf.c
            h = rbf.h
            cs.append(float(c))
            hs.append(float(h))

        data['weights'] = ws
        data['centers'] = cs
        data['widths'] = hs
        data['ax'] = dmp.cs.ax
        data['by'] = dmp.by
        data['dt'] = dmp.dt

        with open(fp, 'w') as file:
            yaml.dump(data, file)

    def loadDMP(self, fp):
        with open(fp, 'r') as file:
            data = yaml.safe_load(file)
        ax = data['ax']
        by = data['by']
        dt = data['dt']
        ws = data['weights']
        rbfData = {'centers':data['centers'], 'widths':data['widths']}
        nRBF = len(ws)
        dmp = DiscreteDMP(nRBF,by,dt,CS(ax,dt),rbfData,ws)
        return dmp

    def saveTraj(self, fp, traj):
        ts = traj[0]
        poses = traj[1]
        fts = traj[2]

        data = {"time":ts, "pose":poses, "force-torques":fts}
        with open(fp, 'w') as file:
            yaml.dump(data, file)

    def loadTraj(self, fp):
        with open(fp, 'r') as file:
            data = yaml.safe_load(file)
        traj = (data["time"], data["pose"], data["force-torques"])
        return(traj)

def ptsEqual(a, b):
    for i in [1,2]:
        #print(len(a[i]))
        #print(6 + i + 2*(i%2-1))
        for j in range(6):# + i + 2*(i%2-1)):
            if not a[i][j] == b[i][j]:
                return False
    return a[0] == b[0]

def dmpEqual(a,b):
    if not a.dt == b.dt:
        return False
    if not a.ax == b.ax:
        return False
    if not a.by == b.by:
        return False
    
    for i in range(len(a.w)):
        if not a.w[i] == b.w[i]:
            return False
        if not a.RBFs[i].c == b.RBFs[i].c:
            return False
        if not a.RBFs[i].h == b.RBFs[i].h:
            return False
        
    return True

if __name__ ==  "__main__":
    ut = DMPUtils()
    bagpath = "rest-looking_down-moveit.bag"
    saveTraj = ut.parseRosbag(bagpath)

    ut.saveTraj("rest-looking_down-moveit.yaml", saveTraj)
    print("Trajectory Saved")
    """
    loadTraj = ut.loadTraj("testTraj2.yaml")
    for i in range(len(saveTraj[0])):
        if not ptsEqual( (saveTraj[0][i], saveTraj[1][i], saveTraj[2][i]), (loadTraj[0][i], loadTraj[1][i], loadTraj[2][i])):
            print('Trajectories are not Equal')
            print('Complete')
            sys.exit()
    print("Successful, trajectories are equal!")

    dmp = DMP_1D(betaY=25.0/4.0, nRBF=5, alphaX=1.0, dt = 0.01)
    ut.saveDMP('testDMP.yaml', dmp)
    print("DMP Saved")
    loadDMP = ut.loadDMP('testDMP.yaml')
    if dmpEqual(dmp, loadDMP):
        print("DMPs are equal, whoo!")
    else:
        print("DMPs are not equal, sad-face")
    print("Complete")
    """