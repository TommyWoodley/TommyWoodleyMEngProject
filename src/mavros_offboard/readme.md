# Run Trajectory Points in Gazebo

## Setup
TODO

## Running a trajectory
Trajectories created by the pybullet simulation are written into the trajectory log file.
Running the trajectory in Gazebo requires some configuration as below:

#### In Terminal 1:

```bash
cd PX4-Autopilot/
DONT_RUN=1 make px4_sitl_default gazebo
source ~/catkin_ws/devel/setup.bash
source Tools/simulation/gazebo-classic/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)/Tools/simulation/gazebo-classic/sitl_gazebo-classic
roslaunch px4 posix_sitl.launch
```
This should start up the gazebo launcher.


#### In Terminal 2:
```bash
roslaunch mavros px4.launch fcu_url:="udp://:14540@192.168.1.36:14557"
```

#### In Terminal 3:
```bash
./QGroundControl.AppImage
```

#### In Terminal 4:
```bash
source ~/catkin_ws/devel/setup.bash
rosrun mavros_offboard sac_traj_setpoint_pendlum.py
```
