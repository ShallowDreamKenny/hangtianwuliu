#!/usr/bin/env python



import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped,PointStamped
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Header
import numpy as np
import math

class CCP:
    # Set publishers
    pub_odom = rospy.Publisher('/odom', Odometry, queue_size=1)
    pub_odom2 = rospy.Publisher('/odom2', Odometry, queue_size=1)

    def __init__(self):
        # init internals
        self.stack_1 = []
        self.stack_2 = []
        self.n = 0
        self.point = PointStamped()
        self.pose1 = PoseStamped()
        self.pose2 = PoseStamped()

        self.car1 = PoseWithCovarianceStamped()
        self.car2 = PoseWithCovarianceStamped()


        self.last_recieved_stamp = None


        # Set the update rate
        rospy.Timer(rospy.Duration(.05), self.timer_callback) # 20hz

        # Set subscribers
        rospy.Subscriber('/clicked_point', PointStamped, self.sub_point)
        rospy.Subscriber('/qingzhou_0/amcl_pose ', PoseWithCovarianceStamped, self.sub_car1)
        rospy.Subscriber('/qingzhou_1/amcl_pose', PoseWithCovarianceStamped, self.sub_car2)

    def sub_car1(self, msg):
        self.car1 = msg

    def sub_car2(self, msg):
        self.car2 = msg
    def sub_point(self, msg):
        # Find the index of the racecar
        if msg != self.point:
            self.point = msg
            self.n += 1
            if self.n<=4:
                if len(self.stack_1) < 4:
                    self.stack_1.append(self.point)
            else:
                if len(self.stack_2) < 4:
                    self.stack_2.append(self.point)


    def step(self):
        while True:
            if self.n >= 8:
                self.pub()

    def pub(self):
        pass




# Start the node
if __name__ == '__main__':
    rospy.init_node("gazebo_odometry_node")
    rospy.spin()

