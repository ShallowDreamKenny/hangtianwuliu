import socket
import threading

import cv2
import queue
import numpy as np
class Send_Img():
    def __init__(self,num):
        self.clinet = socket.socket()
        self.num = num
        self.img_Que = queue.Queue(5)
        self.cap = cv2.VideoCapture(0)

    # 图片传入此处
    def Get_Img(self):
        while self.cap.isOpened():
            try:
                _,img = self.cap.read()
                print(img)
                cv2.imshow('img', img)
                cv2.waitKey(1)
                img = self.cv2bytes(img)
                if self.img_Que.full():
                    self.img_Que.get()
                    self.img_Que.put(img)
                else:
                    self.img_Que.put(img)
            except:
                pass


    def init_Client(self):
        self.clinet = socket.socket()
        self.clinet.connect(('10.41.25.151', 9090))
        print(self.clinet.recv(1024).decode('utf-8'))
        self.count = 1

    def cv2bytes(self,im):
        '''cv2转二进制图片

        :param im: cv2图像，numpy.ndarray
        :return: 二进制图片数据，bytes
        '''
        return np.array(cv2.imencode('.png', im)[1]).tobytes()

    def send_img(self):
        while True:
            msg = self.img_Que.get()
            #print(msg)
            info_size = len(msg)
            print("我已经发了头")
            self.clinet.send((str(info_size) + ':' + str(self.num)).encode('utf-8'))
            res = self.clinet.recv(1024).decode('utf-8')
            if res == 'ok':
                self.clinet.send(msg)
            res2 = self.clinet.recv(1024).decode('utf-8')
            print("我已经完全发了一个")
            if res2 == 'err':
                self.count += 1
            if self.count == 10:
                self.clinet.close()
                raise ValueError

    def Run_Main(self):
        try:
            #threading.Thread(target=self.Get_Img).start()
            self.init_Client()
            self.send_img()

        except Exception:
            self.Run_Main()


Send_Img('map').Run_Main()