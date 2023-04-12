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

建图
step1:打开轻舟 gazebo 仿真世界,在工作空间下新建终端:
roslaunch qingzhou_gazebo qingzhou_bringup.launch

step2:启动 Gmapping 建图,在工作空间下新建终端:
roslaunch qingzhou_mapping cartographer.launch

step3:使用 rqt_robot_steering 控制小车移动,在工作空间下新建终端:
rosrun rqt_robot_steering rqt_robot_steering

step4:保存地图,在工作空间下新建终端:  
  cd ROS/qingzhou/src/qingzhou_mapping/maps/  
  rosrun map_server map_saver -f racemap