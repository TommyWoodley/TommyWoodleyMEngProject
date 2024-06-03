#!/usr/bin/env python
import rospy
import math
import numpy as np
import csv
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import Altitude, ExtendedState, HomePosition, State, WaypointList, PositionTarget, AttitudeTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandTOL
from sensor_msgs.msg import Imu
from controller_msgs.msg import FlatTarget
from pymavlink import mavutil
from std_msgs.msg import Header, Float64
from tf.transformations import quaternion_from_euler
from std_srvs.srv import SetBool

from datalogger import dataLogger
from scipy.spatial.transform import Rotation


class MavrosOffboardSuctionMission():
    """
    Tests flying a path in offboard control by sending position setpoints
    via MAVROS.

    For the test to be successful it needs to reach all setpoints in a certain time.

    FIXME: add flight path assertion (needs transformation from ROS frame to NED)
    """
    GLOBAL_FREQUENCY = 20
    OFFSET = 0.1

    def __init__(self, vy=5):
        # ROS services
        service_timeout = 30
        rospy.loginfo("waiting for ROS services")
        try:
            rospy.wait_for_service('mavros/param/get', service_timeout)
            rospy.wait_for_service('mavros/param/set', service_timeout)
            rospy.wait_for_service('mavros/cmd/arming', service_timeout)
            rospy.wait_for_service('mavros/mission/push', service_timeout)
            rospy.wait_for_service('mavros/mission/clear', service_timeout)
            rospy.wait_for_service('mavros/set_mode', service_timeout)
            rospy.wait_for_service('mavros/cmd/takeoff', service_timeout)
            rospy.wait_for_service('mavros/cmd/land', service_timeout)

            rospy.loginfo("ROS services are up")
        except rospy.ROSException:
            self.fail("failed to connect to services")

        self.vy = vy
        self.vNeeded = 2
        self.droneOrientation = 0

        rospy.loginfo("LOADING WAYPOINTS FROM FILE")
        self.waypoints = self.load_waypoints_from_file(
            "/home/tomwoodley/TommyWoodleyMEngProject/src/mavros_offboard/src/sampled_trajectories/trajectory_1.csv")
        rospy.loginfo("WAYPOINTS: " + str(self.waypoints))

        # mavros service
        self.set_arming_srv = rospy.ServiceProxy('mavros/cmd/arming',
                                                 CommandBool)
        self.set_mode_srv = rospy.ServiceProxy('mavros/set_mode', SetMode)
        self.set_takeoff_srv = rospy.ServiceProxy('mavros/cmd/takeoff', CommandTOL)
        self.set_land_srv = rospy.ServiceProxy('mavros/cmd/land', CommandTOL)

        self.set_trajectory = rospy.ServiceProxy('trajectory', SetBool)
        self.set_waitTrajectory = rospy.ServiceProxy('wait', SetBool)

        # mavros topics
        self.altitude = Altitude()
        self.extended_state = ExtendedState()
        self.imu_data = Imu()
        self.home_position = HomePosition()
        self.local_position = PoseStamped()
        self.local_velocety = TwistStamped()
        self.mission_wp = WaypointList()
        self.state = State()

        self.pos = PoseStamped()
        self.pos_target = PositionTarget()
        self.angle = AttitudeTarget()
        self.flattarget = FlatTarget()
        self.vel = TwistStamped()

        self.pos_setpoint_pub = rospy.Publisher(
            'mavros/setpoint_position/local', PoseStamped, queue_size=1)
        self.pos_target_setpoint_pub = rospy.Publisher(
            'mavros/setpoint_raw/local', PositionTarget, queue_size=1)
        self.att_setpoint_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1)
        self.diff_pub = rospy.Publisher('diff', Float64, queue_size=1)
        self.flatreference_pub = rospy.Publisher("reference/flatsetpoint", FlatTarget, queue_size=1)
        self.vel_setpoint_pub = rospy.Publisher(
            'mavros/setpoint_attitude/cmd_vel', TwistStamped, queue_size=1)

        rospy.wait_for_service("/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        self.pub_pos = True

        self.sub_topics_ready = {
            key: False
            for key in [
                'alt', 'ext_state', 'state', 'imu', 'local_pos', 'local_vel'
            ]
        }

        # ROS subscribers
        self.alt_sub = rospy.Subscriber('mavros/altitude', Altitude,
                                        self.altitude_callback)
        self.ext_state_sub = rospy.Subscriber('mavros/extended_state',
                                              ExtendedState,
                                              self.extended_state_callback)
        self.imu_data_sub = rospy.Subscriber('mavros/imu/data',
                                             Imu,
                                             self.imu_data_callback)
        self.state_sub = rospy.Subscriber('mavros/state', State,
                                          self.state_callback)
        self.local_pos_sub = rospy.Subscriber('mavros/local_position/pose',
                                              PoseStamped,
                                              self.local_position_callback)
        self.local_vel_sub = rospy.Subscriber('mavros/global_position/raw/gps_vel',
                                              TwistStamped,
                                              self.local_velocety_callback)

        # Data Logger object
        self.logData = dataLogger()

    # ----------- LOGGIGNG ---------

    def ros_log_info(self, message):
        rospy.loginfo("WAYPOINT NAV: " + message)

    # ----------- FILE -------------

    def load_waypoints_from_file(self, filename):
        """
        Loads waypoints from a CSV file and returns a list of (x, y, z) tuples.

        Parameters:
        - filename: Name of the CSV file to load.

        Returns:
        - List of (x, y, z) tuples representing the waypoints.
        """
        waypoints = []

        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip the header row

            for row in csvreader:
                x, y, z = map(float, row)
                waypoints.append((x, y, z))

        return waypoints

    # ----------- CALLBACKS -----------

    def altitude_callback(self, data):
        self.altitude = data

        # amsl has been observed to be nan while other fields are valid
        if not self.sub_topics_ready['alt'] and not math.isnan(data.amsl):
            self.sub_topics_ready['alt'] = True

    def extended_state_callback(self, data):
        if self.extended_state.vtol_state != data.vtol_state:
            rospy.loginfo("VTOL state changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_VTOL_STATE']
                [self.extended_state.vtol_state].name, mavutil.mavlink.enums[
                    'MAV_VTOL_STATE'][data.vtol_state].name))

        if self.extended_state.landed_state != data.landed_state:
            rospy.loginfo("landed state changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_LANDED_STATE']
                [self.extended_state.landed_state].name, mavutil.mavlink.enums[
                    'MAV_LANDED_STATE'][data.landed_state].name))

        self.extended_state = data

        if not self.sub_topics_ready['ext_state']:
            self.sub_topics_ready['ext_state'] = True

    def imu_data_callback(self, data):
        self.imu_data = data

        temp = Rotation.from_quat([self.imu_data.orientation.x, self.imu_data.orientation.y,
                                   self.imu_data.orientation.z, self.imu_data.orientation.w])
        self.droneOrientation = temp.as_euler('zyx', degrees=True)

        if not self.sub_topics_ready['imu']:
            self.sub_topics_ready['imu'] = True

    def state_callback(self, data):
        if self.state.armed != data.armed:
            rospy.loginfo("armed state changed from {0} to {1}".format(
                self.state.armed, data.armed))

        if self.state.connected != data.connected:
            rospy.loginfo("connected changed from {0} to {1}".format(
                self.state.connected, data.connected))

        if self.state.mode != data.mode:
            rospy.loginfo("mode changed from {0} to {1}".format(
                self.state.mode, data.mode))

        if self.state.system_status != data.system_status:
            rospy.loginfo("system_status changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_STATE'][
                    self.state.system_status].name, mavutil.mavlink.enums[
                        'MAV_STATE'][data.system_status].name))

        self.state = data

        # mavros publishes a disconnected state message on init
        if not self.sub_topics_ready['state'] and data.connected:
            self.sub_topics_ready['state'] = True

    def local_position_callback(self, data):
        self.local_position = data

        if not self.sub_topics_ready['local_pos']:
            self.sub_topics_ready['local_pos'] = True

    def local_velocety_callback(self, data):
        self.local_velocety = data

        if not self.sub_topics_ready['local_vel']:
            self.sub_topics_ready['local_vel'] = True

    # ----------- GOTO -----------
    def goto_pos_in_time(self, x=0, y=0, z=0, duration=5, prev_x=0, prev_y=0, prev_z=0, writeToDataLogger=True):
        assert duration > 0, "was " + duration

        rate = rospy.Rate(10)  # Hz
        start_time = rospy.Time.now()

        original_x = self.local_position.pose.position.x
        original_y = self.local_position.pose.position.y
        original_z = self.local_position.pose.position.z

        original_distance_x = x - original_x
        original_distance_y = y - original_y
        original_distance_z = z - original_z

        velocity_x = original_distance_x / duration
        velocity_y = original_distance_y / duration
        velocity_z = original_distance_z / duration

        self.pos_target = PositionTarget()
        num_timesteps = duration * 10
        for i in range(num_timesteps):
            if rospy.is_shutdown():
                break
            current_time = rospy.Time.now()
            elapsed_time = current_time - start_time
            self.pos_target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
            self.pos_target.header.stamp = rospy.Time.now()
            self.pos_target.position.x = original_x + velocity_x * min(elapsed_time.to_sec(), duration)
            self.pos_target.position.y = original_y + velocity_y * min(elapsed_time.to_sec(), duration)
            self.pos_target.position.z = original_z + velocity_z * min(elapsed_time.to_sec(), duration)

            self.pos_target.velocity.x = velocity_x
            self.pos_target.velocity.y = velocity_y
            self.pos_target.velocity.z = velocity_z

            self.pos_target_setpoint_pub.publish(self.pos_target)

            if writeToDataLogger:
                interpolated_x = prev_x + ((x - prev_x) / num_timesteps) * (i + 1)
                interpolated_y = prev_y + ((y - prev_y) / num_timesteps) * (i + 1)
                interpolated_z = prev_z + ((z - prev_z) / num_timesteps) * (i + 1)

                self.saveDataToLogData(interpolated_x, interpolated_y, interpolated_z)

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

        current_time = rospy.Time.now()
        time_taken = current_time - start_time
        self.pos.pose.position.x = self.pos_target.position.x
        self.pos.pose.position.y = self.pos_target.position.y
        self.pos.pose.position.z = self.pos_target.position.z

        self.ros_log_info("Time taken: " + str(time_taken.to_sec()))

    def goto_pos(self, x=0, y=0, z=0, writeToDataLogger=True):

        rate = rospy.Rate(10)  # Hz
        reached_pos = False
        self.pos = PoseStamped()

        while not rospy.is_shutdown() and not reached_pos:
            self.pos.header = Header()
            self.pos.header.frame_id = "goto_pos"
            self.pos.pose.position.x = x
            self.pos.pose.position.y = y

            if z >= 0:
                self.pos.pose.position.z = z
            else:
                # in case you use this for the waypoints with negative Z values...
                self.pos.pose.position.z = 0.8

            quaternion = quaternion_from_euler(0.0, 0.0, 0.0)  # roll, pitch, yaw angle
            self.pos.pose.orientation.x = quaternion[0]
            self.pos.pose.orientation.y = quaternion[1]
            self.pos.pose.orientation.z = quaternion[2]
            self.pos.pose.orientation.w = quaternion[3]

            self.pos.header.stamp = rospy.Time.now()
            self.pos_setpoint_pub.publish(self.pos)
            reached_pos = self.is_at_position(x, y, z)

            if writeToDataLogger:
                self.saveDataToLogData(x, y, z)

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    # ----------- HELPERS -----------
    def saveDataToLogData(self, x, y, z):
        self.logData.appendStateData(rospy.Time.now().to_sec(), x, y, z, self.local_position.pose.position.x,
                                     self.local_position.pose.position.y, self.local_position.pose.position.z,
                                     self.local_velocety.twist.linear.x, self.local_velocety.twist.linear.y,
                                     self.local_velocety.twist.linear.z)

    def returnDifference(self, pos):
        diff = ((self.local_position.pose.position.x - pos[0]) ** 2
                + (self.local_position.pose.position.y - pos[1]) ** 2
                + (self.local_position.pose.position.z - pos[2]) ** 2) ** 0.5
        return diff

    def is_at_position(self, x=0, y=0, z=0, printOut=False):
        rospy.logdebug(
            "current position | x:{0:.2f}, y:{1:.2f}, z:{2:.2f}".format(
                self.local_position.pose.position.x, self.local_position.pose.
                position.y, self.local_position.pose.position.z))

        desired = np.array((x, y, z))
        pos = np.array((self.local_position.pose.position.x,
                        self.local_position.pose.position.y,
                        self.local_position.pose.position.z))

        if printOut:
            rospy.loginfo("goto x:{0:.4f}, y:{1:.4f}, z:{2:.4f}  ".format(desired[0], desired[1], desired[2])
                          + "|  current x:{0:.4f}, y:{1:.4f}, z:{2:.4f}  ".format(pos[0], pos[1], pos[2])
                          + "| diff:{0:.4f}".format(np.linalg.norm(desired - pos)))
        return np.linalg.norm(desired - pos) < self.OFFSET

    # ----------- FLIGHT PATH METHODS -----------

    def navigate_to_starting_position(self, rate, initX, initY, initZ, last_req):
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'

        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True

        while not rospy.is_shutdown():
            if self.state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                if self.set_mode_client.call(offb_set_mode).mode_sent:
                    rospy.loginfo("OFFBOARD enabled")
                last_req = rospy.Time.now()
            else:
                if not self.state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                    if self.arming_client.call(arm_cmd).success:
                        rospy.loginfo("Vehicle armed")
                    last_req = rospy.Time.now()

            self.pos_setpoint_pub.publish(self.pos)

            if self.is_at_position(initX, initY, initZ, printOut=False):
                rospy.loginfo("INITIAL POSITION REACHED")
                break
            rate.sleep()
        return last_req

    # ----------- HOVER -----------

    def hover_at_current_pos(self, time):
        num_waiting_steps = time * self.GLOBAL_FREQUENCY
        rate = rospy.Rate(self.GLOBAL_FREQUENCY)

        pose = PoseStamped()
        pose.pose.position.x = self.pos.pose.position.x
        pose.pose.position.y = self.pos.pose.position.y
        pose.pose.position.z = self.pos.pose.position.z

        # sent point until time is over
        self.ros_log_info("About to begin waiting for " + str(time) + "seconds.")
        for i in range(num_waiting_steps):
            if rospy.is_shutdown():
                break

            self.pos_setpoint_pub.publish(pose)
            # self.saveDataToLogData(self.pos.pose.position.x, self.pos.pose.position.y, self.pos.pose.position.z)
            rate.sleep()

        self.ros_log_info("Waiting Over")

    def confirm_next_stage(self, message, hover):
        if rospy.has_param('mission_confirm'):
            rospy.delete_param('mission_confirm')
        self.ros_log_info(message + " (set ROS parameter 'mission_confirm' to 'confirm' or 'stop')")
        while not rospy.is_shutdown():
            if rospy.has_param('mission_confirm'):
                user_input = rospy.get_param('mission_confirm')
                if isinstance(user_input, str):
                    user_input = user_input.strip().lower()
                    if user_input == 'confirm':
                        rospy.delete_param('mission_confirm')
                        return True
                    elif user_input == 'stop':
                        rospy.delete_param('mission_confirm')
                        return False
                    else:
                        rospy.delete_param('mission_confirm')
                        self.ros_log_info("Invalid input. Please set 'mission_confirm' to 'confirm' or 'stop'.")
                else:
                    rospy.delete_param('mission_confirm')
                    self.ros_log_info(
                        "Invalid input type. Please set 'mission_confirm' to 'confirm' or 'stop': " + str(user_input))
            if hover:
                pose = PoseStamped()
                pose.pose.position.x = self.pos.pose.position.x
                pose.pose.position.y = self.pos.pose.position.y
                pose.pose.position.z = self.pos.pose.position.z
                self.pos_setpoint_pub.publish(pose)

            rospy.sleep(1)

    # ----------- FLIGHT PATH METHODS -----------
    def run_full_mission(self):
        # Setpoint publishing MUST be faster than 2Hz
        rate = rospy.Rate(20)

        # Wait for Flight Controller connection
        while not rospy.is_shutdown() and not self.state.connected:
            rate.sleep()

        # Initially take off 2m in the air and wait
        initX = self.local_position.pose.position.x
        initY = self.local_position.pose.position.y
        initZ = self.local_position.pose.position.z + 1  # take off 1m

        self.pos.pose.position.x = initX
        self.pos.pose.position.y = initY
        self.pos.pose.position.z = initZ

        self.startup_mission(rate)

        if not self.confirm_next_stage("Confirm Drone Takeoff", hover=False):
            return

        self.ros_log_info("TAKEOFF")
        last_req = self.navigate_to_starting_position(rate, initX, initY, initZ, last_req=rospy.Time.now())
        self.ros_log_info("TAKEOFF ACHIEVED")

        self.ros_log_info("HOVER @ TAKEOFF POSITION 5s")
        self.hover_at_current_pos(time=3)

        if not self.confirm_next_stage("Confirm Drone Move to Starting", hover=True):
            return

        self.ros_log_info("NAVIGATE TO STARTING POSITION")
        x_start, y_start, z_start = self.waypoints[0]
        self.goto_pos(x=x_start, y=y_start, z=z_start, writeToDataLogger=False)
        self.ros_log_info("REACHED STARTING POSITION")

        self.ros_log_info("HOVER @ STARTING POSITION 10s")
        self.hover_at_current_pos(time=5)

        if not self.confirm_next_stage("Confirm Start Trajectory", hover=True):
            return

        self.ros_log_info("STARTING TRAJECTORY")

        waypoints = self.waypoints
        time_between_waypoint = 1

        for index, (x, y, z) in enumerate(waypoints):
            self.ros_log_info("HEADING TO WAYPOINT " + str(index))
            prev_index = index - 1 if index - 1 >= 0 else 0
            prev_x, prev_y, prev_z = waypoints[prev_index]
            self.goto_pos_in_time(x, y, z, time_between_waypoint, prev_x, prev_y, prev_z)

        self.ros_log_info("TRAJECTORY ENDED")

        # go to original pos
        rospy.loginfo("---- LAND ----")
        self.goto_pos(initX, initY, initZ, writeToDataLogger=False)
        land_set_mode = SetModeRequest()
        land_set_mode.custom_mode = 'AUTO.LAND'

        last_req = rospy.Time.now()

        while not rospy.is_shutdown() and self.state.armed:
            if self.state.mode != "AUTO.LAND" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                if self.set_mode_client.call(land_set_mode).mode_sent:
                    rospy.loginfo("AUTO.LAND enabled")
                last_req = rospy.Time.now()
            self.pos_setpoint_pub.publish(self.pos)
            rate.sleep()

    def startup_mission(self, rate):
        # Send a few setpoints before starting
        for i in range(100):
            if rospy.is_shutdown():
                break

            self.pos_setpoint_pub.publish(self.pos)
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('offboard_mission_node')
    suction_mission = MavrosOffboardSuctionMission()
    suction_mission.run_full_mission()

    suction_mission.logData.saveAll()
    suction_mission.logData.plotFigure()
    rospy.loginfo("huhu")
