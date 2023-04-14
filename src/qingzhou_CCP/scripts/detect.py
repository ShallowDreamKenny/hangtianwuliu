#!/usr/bin/env python3
"""
# File       : preprocess_vector.py
# Time       ：12/22/22 8:23 AM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import argparse
import sys

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from numpy import random
from sensor_msgs.msg import Image

from src.qingzhou_CCP.yolov7.detect_car import Detector


"""
-------------------------------------
    # Args Init
-------------------------------------
"""
def args_set():
    """
    Initialization Parameter
    Returns:
        Args:
    """
    #Yolo Part
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default='true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    return parser.parse_args()

def timer_callback1(event):
    global poseC, pose
    try:
        if Vec.get_vector(Vec.color_image, "car1"):
            pose.header = poseC.header
            # pose.header.frame_id = "map"
            pose.pose = poseC.pose.pose
            pose.pose.position.x = pose.pose.position.x+1
            print(pose)
            pub_target.publish(pose)
            return True

    except:
        print("data fusion faild")
    return False

def timer_callback2(event):
    try:
        Vec.send_img(img,"map",Vec.clinet4)
    except:
        print("map tr faild")

poseC = PoseWithCovarianceStamped()
pose = PoseStamped()
def callback(msg):
    global poseC
    poseC = msg

if __name__ == '__main__':
    args = args_set()
    rospy.init_node('get_image', anonymous=True)
    rospy.Subscriber("/qingzhou_1/amcl_pose", PoseWithCovarianceStamped, callback)

    pub_target = rospy.Publisher('/qingzhou_2/move_base_simple/goal', PoseStamped, queue_size=1)
    img = cv2.imread("res.png")
    # poseC = PoseWithCovarianceStamped()
    rospy.Timer(rospy.Duration(1), timer_callback2)
    Vec = Detector(args)

    while True:
        if timer_callback1(1):
            print("实施围捕")

