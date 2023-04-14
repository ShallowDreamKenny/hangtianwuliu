import socket
import threading
import queue
import cv2
import numpy as np

class Socket_Img():
    def __init__(self):
        self.server = socket.socket()
        self.Que1 = queue.Queue(5)
        self.Que2 = queue.Queue(5)
        self.Que3 = queue.Queue(5)
        self.Quemap = queue.Queue(5)

    def init_Server(self):
        self.server = socket.socket()
        self.server.bind(('127.0.0.1', 9090))
        self.server.listen(5)

    def get_Client(self):
        while True:
            gclient, addr = self.server.accept()
            print(addr)
            gclient.send("欢迎访问".encode('utf-8'))
            try:
                threading.Thread(target=self.img_get, args=(gclient,), daemon=True).start()
            except Exception:
                continue

    def img_get(self, gclient):
        try:
            while True:
                # 接收头
                info_size, info_who = gclient.recv(1024).decode('utf-8').split(':')
                # print("数据长度:",info_size)
                # 返还ok
                # print("是这个ok问题?1")
                gclient.send("ok".encode('utf-8'))
                # print("是这个ok问题?")
                # 循环接收长数据
                info = b''
                if int(info_size) % 1024 == 0:
                    for i in range((int(info_size) // 1024)):
                        info += gclient.recv(1024)
                else:
                    for i in range((int(info_size) // 1024 + 1)):
                        info += gclient.recv(1024)

                if len(info) == int(info_size):
                    gclient.send("ok".encode('utf-8'))
                else:
                    gclient.send("err".encode('utf-8'))
                #print(info.decode("utf-8"))
                self.save_info(info_who, info)
                print("接收了一条信息 存储完毕")

        except Exception:
            gclient.close()

    def save_info(self, num, info):
        if num == '1':
            if self.Que1.full():
                self.Que1.get()
            else:
                self.Que1.put(info)
        elif num == '2':
            if self.Que2.full():
                self.Que2.get()
            else:
                self.Que2.put(info)
        elif num == '3':
            if self.Que3.full():
                self.Que3.get()
            else:
                self.Que3.put(info)
        elif num == 'map':
            if self.Quemap.full():
                self.Quemap.get()
            else:
                self.Quemap.put(info)
        else:
            raise ValueError

    def bytes2cv(self,im):
        '''二进制图片转cv2

        :param im: 二进制图片数据，bytes
        :return: cv2图像，numpy.ndarray
        '''
        return cv2.imdecode(np.array(bytearray(im), dtype='uint8'), cv2.IMREAD_UNCHANGED)  # 从二进制图片数据中读取

    def Read_info(self, num):
        res = None
        if num == '1':
            if self.Que1.empty():
                res = None
            else:
                res = self.Que1.get()
        elif num == '2':
            if self.Que2.empty():
                res = None
            else:
                res = self.Que2.get()
        elif num == '3':
            if self.Que3.empty():
                res = None
            else:
                res = self.Que3.get()
        elif num == 'map':
            if self.Quemap.empty():
                res = None
            else:
                res = self.Quemap.get()
        return res

    def Run_Main(self):
        try:
            self.init_Server()
            self.get_Client()
        except Exception:
            self.Run_Main()


# Server = Socket_Img()
# threading.Thread(target=Server.Run_Main,daemon=True).start()
# while True:
#     img = Server.bytes2cv(Server.Read_info('1'))
#     cv2.imshow('img_get',img)
#     cv2.waitKey(50)
#     pass