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
from hwt_data.msg import Hwt_ht_basic
from numpy import random
from sensor_msgs.msg import Image

# from Environment_preprocessing.test_yolo import plot_frame
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

sys.path.append('../yolov7')


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

class Detector():
    def __init__(self, args):
        self.args = args
        self.color_image, self.depth_image = None, None
        self.color_image2, self.depth_image2 = None, None
        self.old_img_b, self.old_img_h, self.old_img_w = None, None, None

        rospy.Subscriber("qingzhou_1/camera_link/image_raw", Image, self.callback)
        rospy.Subscriber("qingzhou_0/camera_link/image_raw", Image, self.callback1)
        # rospy.Timer(rospy.Duration(.5), self.timer_callback)  # 2hz
        self.r = rospy.Rate(10)  # 10Hz
        # Init Model
        self.model, self.imgsz, self.device, self.half, self.view_img, \
            self.names, self.colors, self.stride = self.init_model(self.args)

    """
    -------------------------------------
        # detect model init and get
    -------------------------------------
    """

    def init_model(self, args):
        # Init Yolo Model
        weights, view_img, save_txt, imgsz, trace = args.weights, args.view_img, \
            args.save_txt, args.img_size, not args.no_trace
        set_logging()
        device = select_device(args.device)
        half = device.type != 'cpu'

        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, args.img_size)

        if half:
            model.half()  # to FP16

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1
        return model, imgsz, device, half, view_img, names, colors, stride

    def plot_frame(self, pred, img1, img, names, colors, name):
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img1.shape[2:], det[:, :4], img.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=1)

            cv2.imshow(name, img)
            cv2.waitKey(1)  # 1 millisecond

    def detect(self, img, name):
        img1 = letterbox(img, self.imgsz, stride=self.stride)[0]
        img1 = img1[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img1 = np.ascontiguousarray(img1)

        img1 = torch.from_numpy(img1).to(self.device)
        img1 = img1.self.half() if self.half else img1.float()  # uint8 to fp16/32
        img1 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img1.ndimension() == 3:
            img1 = img1.unsqueeze(0)

        if self.device.type != 'cpu' and (
                self.old_img_b != img1.shape[0] or self.old_img_h != img1.shape[2] or self.old_img_w != img1.shape[3]):
            self.old_img_b = img1.shape[0]
            self.old_img_h = img1.shape[2]
            self.old_img_w = img1.shape[3]
            for i in range(3):
                self.model(img1, augment=self.args.augment)[0]

        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img1, augment=self.args.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres, classes=self.args.classes,
                                   agnostic=self.args.agnostic_nms)
        # print(pred)

        if self.view_img:
            self.plot_frame(pred, img1, img, self.names, self.colors, name)

        return pred


    """
    -------------------------------------
        # Image Get
    -------------------------------------
    """

    def callback(self, data1):
        bridge = CvBridge()
        self.color_image = bridge.imgmsg_to_cv2(data1, 'bgr8')

    def callback1(self, data2):
        bridge = CvBridge()
        self.color_image2 = bridge.imgmsg_to_cv2(data2, 'bgr8')




    """
    -------------------------------------
        # Data fusion
    -------------------------------------
    """

    def get_vector(self, image, name):
        pred = self.detect(image, name)
        pred = pred[0].numpy()
        if pred.shape[0] != 0:
            print(pred)
            return True
        #print(pred)
        return False


"""
-------------------------------------
    # test
-------------------------------------
"""
if __name__ == '__main__':
    args = args_set()
    rospy.init_node('get_image', anonymous=True)
    Vec = Detector(args)

    while True:
        try:
            if Vec.get_vector(Vec.color_image, "car1"):
                print("car 1 !!!!")
            if Vec.get_vector(Vec.color_image2, "car2"):
                print("car 2 !!!!")
        except:
            print("detect wrone")
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            rospy.signal_shutdown("shut_down")
            break

    # rospy.spin()

