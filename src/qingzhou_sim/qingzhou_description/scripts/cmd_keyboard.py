#!/usr/bin/env python

from cmd import Cmd
import rospy
import math
import curses
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayDimension
from geometry_msgs.msg import Twist

msg = """
Control The qingzhou Robot!
---------------------------
Moving around:
        W     
   A    S    D

W/S : Forward and backward
A/D : Turn left and turn right

CTRL-C to quit
"""

# GLOBAL KEYS' DICTIONARY: Associates a keyboard key to a vector (x,y,z,theta)
vel_bindings = {
    'w': [3, 3, 3, 3],      # Forward 2m/s
    's': [-3, -3, -3, -3]   # Backward 2m/s
}

steer_bindings = {
    'a': 0.17,  # 10grad anti-clock-wise
    'd': -0.17  # 10grad clock-wise
}

# GLOBAL: MODEL GEOMETRY
L = 0.3  # Distance between wheels axes
W = 0.3  # Distance between wheels


# FUNCTION start_curses(): Start a curses terminal app
def start_curses():
    app = curses.initscr()  # Create a terminal window
    curses.noecho()         # Makes input invisible
    app.addstr(msg)         # Print the start message
    return app


# MAIN FUNCTION
def move():
    # Defining the topic to publish on
    pub_vel = rospy.Publisher(
        '/qingzhou/ackermann_steering_controller/cmd_vel', Twist, queue_size=5)

    # Defining the name of the node represented by this script
    rospy.init_node('ackrm_robot_teleop', anonymous=True)

    # Setting the rate of publications at 20hz
    rate = rospy.Rate(20)
    # Initializing velocity msg
    theta = 0.0
    linear = 0.0

    # Starting and configuring Curses application
    app = start_curses()

    # MAIN WHILE LOOP
    while not rospy.is_shutdown():

        # Reading the pressed key from the curses app
        key = app.getkey()

        # Selecting correct speed and angle based on the pressed key
        if key in vel_bindings.keys():
            if key == 'w':
                linear += 0.2
            elif key == 's':
                linear -= 0.2

            if linear > 1:
                linear = 1
            elif linear < -1:
                linear = -1

        elif key in steer_bindings.keys():
            theta = theta + steer_bindings[key]
            if theta > math.pi/8:
                theta = math.pi/8
            elif theta < -math.pi/8:
                theta = -math.pi/8

        else:
            # Incorrect key => Robot stop
            linear = 0.0
            theta = 0.0
            # q => exit from loop
            if (key == 'q'):
                curses.endwin()  # End Curses application
                break

        # Calculating wheel's steering angle
     
        cmd = Twist()
        cmd.angular.z = theta
        cmd.linear.x = linear
        # Publishing the messages
        pub_vel.publish(cmd)

        # Pause the loop
        rate.sleep()


# SIMPLE SCRIPT MAIN WITH ERROR EXCEPTION
if __name__ == '__main__':
    try:
        move()
    except rospy.ROSInterruptException:
        pass
    except curses.error:
        curses.endwin()