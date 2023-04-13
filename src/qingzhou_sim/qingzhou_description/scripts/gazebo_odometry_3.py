#!/usr/bin/env python



import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, Transform, TransformStamped
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Header
import numpy as np
import math

class OdometryNode:
    # Set publishers
    pub_odom = rospy.Publisher('/odom', Odometry, queue_size=1)
    pub_odom2 = rospy.Publisher('/odom2', Odometry, queue_size=1)
    pub_odom3 = rospy.Publisher('/odom3', Odometry, queue_size=1)

    def __init__(self):
        # init internals
        self.last_received_pose = Pose()
        self.last_received_twist = Twist()
        self.last_received_pose2 = Pose()
        self.last_received_twist2 = Twist()
        self.last_received_pose3 = Pose()
        self.last_received_twist3 = Twist()
        self.last_recieved_stamp = None
        self.last_recieved_stamp2 = None
        self.last_recieved_stamp3 = None

        # Set the update rate
        rospy.Timer(rospy.Duration(.05), self.timer_callback) # 20hz

        # Set subscribers
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.sub_robot_pose_update)

    def sub_robot_pose_update(self, msg):
        # Find the index of the racecar
        try:
            arrayIndex = msg.name.index('/qingzhou_0')
            arrayIndex2 = msg.name.index('/qingzhou_1')
            arrayIndex3 = msg.name.index('/qingzhou_2')
        except ValueError as e:
            # Wait for Gazebo to startup
            #print("1")
            pass
        else:
            # Extract our current position information
            self.last_received_pose = msg.pose[arrayIndex]
            self.last_received_twist = msg.twist[arrayIndex]
            self.last_received_pose2 = msg.pose[arrayIndex2]
            self.last_received_twist2 = msg.twist[arrayIndex2]
            self.last_received_pose3 = msg.pose[arrayIndex3]
            self.last_received_twist3 = msg.twist[arrayIndex3]
            #print(self.last_received_twist2)
        self.last_recieved_stamp = rospy.Time.now()
        self.last_recieved_stamp2 = rospy.Time.now()
        self.last_recieved_stamp3 = rospy.Time.now()

    def timer_callback(self, event):
        if self.last_recieved_stamp is None:
            return

        cmd = Odometry()
        cmd.header.stamp = self.last_recieved_stamp
        cmd.header.frame_id = 'qingzhou_0/odom'
        cmd.child_frame_id = 'qingzhou_0/base_link'
        cmd.pose.pose = self.last_received_pose
        cmd.twist.twist = self.last_received_twist
        cmd.pose.covariance =[1e-3, 0, 0, 0, 0, 0,
                              0, 1e-3, 0, 0, 0, 0,
                              0, 0, 1e6, 0, 0, 0,
                              0, 0, 0, 1e6, 0, 0,
                              0, 0, 0, 0, 1e6, 0,
                              0, 0, 0, 0, 0, 1e3]

        cmd.twist.covariance = [1e-9, 0, 0, 0, 0, 0,
                                0, 1e-3, 1e-9, 0, 0, 0,
                                0, 0, 1e6, 0, 0, 0,
                                0, 0, 0, 1e6, 0, 0,
                                0, 0, 0, 0, 1e6, 0,
                                0, 0, 0, 0, 0, 1e-9]
        cmd2 = Odometry()
        cmd2.header.stamp = self.last_recieved_stamp2
        cmd2.header.frame_id = 'qingzhou_1/odom'
        cmd2.child_frame_id = 'qingzhou_1/base_link'
        cmd2.pose.pose = self.last_received_pose2
        cmd2.twist.twist = self.last_received_twist2
        cmd2.pose.covariance =[1e-3, 0, 0, 0, 0, 0,
                               0, 1e-3, 0, 0, 0, 0,
                               0, 0, 1e6, 0, 0, 0,
                               0, 0, 0, 1e6, 0, 0,
                               0, 0, 0, 0, 1e6, 0,
                               0, 0, 0, 0, 0, 1e3]

        cmd2.twist.covariance = [1e-9, 0, 0, 0, 0, 0,
                                 0, 1e-3, 1e-9, 0, 0, 0,
                                 0, 0, 1e6, 0, 0, 0,
                                 0, 0, 0, 1e6, 0, 0,
                                 0, 0, 0, 0, 1e6, 0,
                                 0, 0, 0, 0, 0, 1e-9]

        cmd3 = Odometry()
        cmd3.header.stamp = self.last_recieved_stamp3
        cmd3.header.frame_id = 'qingzhou_2/odom'
        cmd3.child_frame_id = 'qingzhou_2/base_link'
        cmd3.pose.pose = self.last_received_pose3
        cmd3.twist.twist = self.last_received_twist3
        cmd3.pose.covariance =[1e-3, 0, 0, 0, 0, 0,
                               0, 1e-3, 0, 0, 0, 0,
                               0, 0, 1e6, 0, 0, 0,
                               0, 0, 0, 1e6, 0, 0,
                               0, 0, 0, 0, 1e6, 0,
                               0, 0, 0, 0, 0, 1e3]

        cmd3.twist.covariance = [1e-9, 0, 0, 0, 0, 0,
                                 0, 1e-3, 1e-9, 0, 0, 0,
                                 0, 0, 1e6, 0, 0, 0,
                                 0, 0, 0, 1e6, 0, 0,
                                 0, 0, 0, 0, 1e6, 0,
                                 0, 0, 0, 0, 0, 1e-9]


        self.pub_odom3.publish(cmd3)
        self.pub_odom2.publish(cmd2)
        self.pub_odom.publish(cmd)


# Start the node
if __name__ == '__main__':
    rospy.init_node("gazebo_odometry_node")
    node = OdometryNode()
    rospy.spin()

