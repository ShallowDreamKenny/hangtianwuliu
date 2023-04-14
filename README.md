# qingzhou
sudo apt install ros-melodic-ackermann-steering-controller      
sudo apt-get install ros-(your ros version)-cartographer*     
sudo apt-get install -y 
cmake 
g++ 
git 
google-mock 
libboost-all-dev 
libcairo2-dev 
libeigen3-dev libgflags-dev 
libgoogle-glog-dev 
liblua5.2-dev 
libsuitesparse-dev 
libwebp-dev 
ninja-build 
protobuf-compiler 
python-sphinx

### 多车建图
#### step1:打开轻舟 gazebo 仿真世界,在工作空间下新建终端:
    roslaunch qingzhou_gazebo qingzhou_bringup_multi2.launch

#### step2:启动 Gmapping 建图,在工作空间下新建终端:  
    roslaunch qingzhou_mapping gmapping_multi.launch

#### step3:使用 rqt_robot_steering 控制小车移动,在工作空间下新建终端:  
    rosrun rqt_robot_steering rqt_robot_steering  
    rosrun rqt_robot_steering rqt_robot_steering

#### step4:保存地图,在工作空间下新建终端:  
    cd ROS/qingzhou/src/qingzhou_mapping/maps/  
    rosrun map_server map_saver -f racemap /map:=/qingzhou_1/map  
    rosrun map_server map_saver -f racemap /map:=/qingzhou_0/map  
#### step5: 融合地图节点
    cd qingzhou/src/qingzhou_mapping/scripts/  
    python3 merge.py

### 多车导航
####  Step1：启动3小车仿真  
    roslaunch qingzhou_gazebo qingzhou_bringup_multi3.launch
#### Step2：启动小车导航  
    roslaunch qingzhou_nav multi_nav_bringup.launch  
#### Step3：启动目标检测、目标围堵、上位机通信节点  
    cd qingzhou/src/qingzhou_CCP/scripts/  
    python3 detect.py
    打开上位机显示程序
#### Step4: 使用全覆盖路径规划搜索目标或手动导航搜索目标
 ###### 全覆盖路径规划(待完善)
    cd qingzhou/src/qingzhou_CCP/src/  
    python3 transform_pub.py
 ###### 手动导航
    手动发布节点给各个小车，小车便会在地图内进行导航及巡查工作
