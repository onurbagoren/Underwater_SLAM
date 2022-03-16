from matplotlib import projections
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from mpl_toolkits.mplot3d import Axes3D


import matplotlib.pyplot as plt
import sys
import pandas as pd
from scipy.spatial.transform import Rotation as R

class SonarPlotter():

    def __init__(self):
        self.ranges = None
        self.angle = None

        self.DATADIR = f'{sys.path[0]}/../data/full_dataset/'

    def odom_callback(self, data):
        position = data.pose.pose.position
        orientation = data.pose.pose.orientation
        [x, y, z, w] = [orientation.x, orientation.y, orientation.z, orientation.w]
        euler = R.from_quat([x, y, z, w]).as_euler('xyz', degrees=True)

    def seaking_ros_callback(self, data):
        header = data.header
        angle_min = data.angle_min
        angle_max = data.angle_max
        angle_increment = data.angle_increment
        self.ranges = data.ranges
        range_min = data.range_min
        range_max = data.range_max
        scan_time = data.scan_time
        time_increment = data.time_increment

    def listener(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber('/sonar_seaking_ros', LaserScan,
                         self.seaking_ros_callback)
        rospy.Subscriber('/odometry', Odometry, self.odom_callback)
        rospy.spin()

    def plot_sonar(self):
        seaking_data = pd.read_csv(self.DATADIR + 'sonar_seaking.txt')
        print(seaking_data['field.angle_grad'])
    
    def plot_odometry(self):
        odom_data = pd.read_csv(self.DATADIR + 'odometry.txt')
        x = odom_data['field.pose.pose.position.x']
        y = odom_data['field.pose.pose.position.y']
        z = odom_data['field.pose.pose.position.z']

        # 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

def main():
    sp = SonarPlotter()
    sp.plot_odometry()


if __name__ == '__main__':
    main()
