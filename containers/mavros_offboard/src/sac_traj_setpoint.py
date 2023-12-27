#!/usr/bin/env python3
from __future__ import division

import rospy
import math
import numpy as np
import time
from geometry_msgs.msg import PoseStamped, Quaternion, Point, Vector3, Twist, TwistStamped
from mavros_msgs.msg import ParamValue
from mavros_msgs.msg import Altitude, ExtendedState, HomePosition, ParamValue, State, \
                            WaypointList, PositionTarget
from mavros_msgs.srv import CommandBool, ParamGet, ParamSet, SetMode, SetModeRequest, WaypointClear, \
                            WaypointPush, CommandTOL
from sensor_msgs.msg import NavSatFix, Imu
from controller_msgs.msg import FlatTarget
#from mavros_test_common import MavrosTestCommon
from pymavlink import mavutil
from six.moves import xrange
from std_msgs.msg import Header, Float64
from threading import Thread
from tf.transformations import quaternion_from_euler
from multiprocessing import Value
from ctypes import c_int
from collections import deque
from std_srvs.srv import SetBool, SetBoolRequest

class MavrosOffboardSuctionMission():
    """
    Tests flying a path in offboard control by sending position setpoints
    via MAVROS.

    For the test to be successful it needs to reach all setpoints in a certain time.

    FIXME: add flight path assertion (needs transformation from ROS frame to NED)
    """

    def __init__(self, offset = 0.1, vy = 5):
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
            
            #rospy.wait_for_service('/trajectory', service_timeout)
            #rospy.wait_for_service('/wait', service_timeout)
            
            rospy.loginfo("ROS services are up")
        except rospy.ROSException:
            self.fail("failed to connect to services")

        self.dp = None
        self.vy = vy
        try:
            ## Enter full path of the waypointn txt file here
            self.dp = np.loadtxt("datapoint235.txt", dtype='float', delimiter=' ', skiprows=2)
            rospy.loginfo("Read datapoint file.")
            self.vy *= self.dp[-1][4] # correct the direction based on the last entry of the waypoint file
        except Exception as e:
            self.fail("Failed to read datapoint")

        
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
        self.mission_wp = WaypointList()
        self.state = State()
        self.offset = offset
        
        self.pos = PoseStamped()
        self.pos_target = PositionTarget()
        self.flattarget = FlatTarget()
        self.vel = TwistStamped()

        self.pos_setpoint_pub = rospy.Publisher(
            'mavros/setpoint_position/local', PoseStamped, queue_size=1)
        self.pos_target_setpoint_pub = rospy.Publisher(
            'mavros/setpoint_raw/local', PositionTarget, queue_size=1)
        self.diff_pub = rospy.Publisher('diff', Float64, queue_size=1)
        self.flatreference_pub = rospy.Publisher("reference/flatsetpoint", FlatTarget, queue_size=1)
        self.vel_setpoint_pub = rospy.Publisher(
            'mavros/setpoint_attitude/cmd_vel', TwistStamped, queue_size=1)
        self.pub_pos = True

        self.sub_topics_ready = {
            key: False
            for key in [
                'alt', 'ext_state', 'state', 'imu', 'local_pos'
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


    #
    # Helper methods
    #
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
            


        
    def goto_pos(self, x=0, y=0, z=0):
        from tf.transformations import quaternion_from_euler

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

            quaternion = quaternion_from_euler(0.0, 0.0, 0.0) # roll, pitch, yaw angle
            self.pos.pose.orientation.x = quaternion[0]
            self.pos.pose.orientation.y = quaternion[1]
            self.pos.pose.orientation.z = quaternion[2]
            self.pos.pose.orientation.w = quaternion[3]

            self.pos.header.stamp = rospy.Time.now()
            self.pos_setpoint_pub.publish(self.pos)
            reached_pos = self.is_at_position(self.offset, x, y, z)

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass



    def send_pos_target(self):
        import math
        rate = rospy.Rate(10)  # Hz
        count = 0
        self.pos_target = PositionTarget()
                
        while not rospy.is_shutdown() and count < self.dp.shape[0]-2:
            self.pos_target.header = Header()
            self.pos_target.header.frame_id = "trajectory_pos"
            self.pos_target.header.stamp = rospy.Time.now()
            self.pos_target.type_mask = PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ + \
                                        PositionTarget.FORCE + PositionTarget.IGNORE_YAW +  \
                                        PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ
            self.pos_target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

            self.pos_target.position.x = self.dp[count][0]
            self.pos_target.position.y = self.dp[count][1]
            self.pos_target.position.z = self.dp[count][2]

            self.pos_target.yaw_rate = 0.0
            self.pos_target_setpoint_pub.publish(self.pos_target)

            reached_pos = self.is_at_position(self.offset, self.dp[count][0], self.dp[count][1], self.dp[count][2])
            count+=1
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.loginfo("Position SP completes. Passing on to Velocity SP!")



    def send_vel(self):
        rate = rospy.Rate(50)  # Hz
        reached_pos = False
        central = False
        end_pos = (self.dp[-1][0] , self.dp[-1][1], self.dp[-1][2])
        self.pos_target = PositionTarget()
        
        while not reached_pos:
            '''
            self.vel.header.frame_id = "vel"
            self.vel.twist.linear.x = 0
            self.vel.twist.linear.y = self.vy
            self.vel.twist.linear.z = 0
            self.vel.twist.angular.x = 0
            self.vel.twist.angular.y = 0
            self.vel.twist.angular.z = 0
            self.vel.header.stamp = rospy.Time.now()
            self.vel_setpoint_pub.publish(self.vel)
            '''
            
            self.pos_target.header = Header()
            self.pos_target.header.frame_id = "velocity_pos"
            self.pos_target.header.stamp = rospy.Time.now()
            self.pos_target.type_mask = PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ + \
                                        PositionTarget.FORCE + PositionTarget.IGNORE_YAW_RATE +  \
                                        PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ
            self.pos_target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
            
            ## Fusion of acceleration + vel doesn't work
            ##self.pos_target.type_mask = PositionTarget.FORCE + PositionTarget.IGNORE_YAW_RATE +  \
            ##                            PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ

            self.pos_target.velocity.x = 0
            self.pos_target.velocity.y = self.vy
            self.pos_target.velocity.z = 0
            
            #self.pos_target.acceleration_or_force.x = 0
            if not central:
                self.pos_target.velocity.y = self.vy
            else:
                self.pos_target.velocity.y = self.vy * -1.0
            #self.pos_target.acceleration_or_force.z = 0
            
            self.pos_target.yaw = 0
            self.pos_target_setpoint_pub.publish(self.pos_target)
            
            central = abs(self.dp[-2][1] - self.local_position.pose.position.y) > abs(self.dp[-2][1] - self.dp[-1][1])/2 - self.offset
                
            
            reached_pos = abs(self.dp[-2][1] - self.local_position.pose.position.y) > abs(self.dp[-2][1] - self.dp[-1][1]) - self.offset
            
            print("Vel: ", self.local_position.pose.position.y)
            print(central)

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        
        # test of sending a setpoint with zero vy after the loop. May not work well as it needs a constant stream for a few second. Here only does once.
        self.pos_target.header = Header()
        self.pos_target.header.frame_id = "velocity_pos"
        self.pos_target.header.stamp = rospy.Time.now()
        self.pos_target.type_mask = PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ + \
                                        PositionTarget.FORCE + PositionTarget.IGNORE_YAW_RATE +  \
                                        PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ
        self.pos_target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

        self.pos_target.velocity.x = 0
        self.pos_target.velocity.y = 0
        self.pos_target.velocity.z = 0
        self.pos_target.yaw = 0
        self.pos_target_setpoint_pub.publish(self.pos_target)
                
        rospy.loginfo("End!")


    def send_pos_raw(self):
        import math
        rate = rospy.Rate(2)  # Hz
        wn = 1.0
        r = 2.0
        count = 0
        self.pos_target = PositionTarget()
        
        while not rospy.is_shutdown() and count < self.dp.shape[0]:
            self.pos_target.header = Header()
            self.pos_target.header.frame_id = "datapoint"
            self.pos_target.header.stamp = rospy.Time.now()
            self.pos_target.type_mask = PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ + \
                                        PositionTarget.FORCE + PositionTarget.IGNORE_YAW_RATE # + \
                                        #PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ 
            self.pos_target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

            self.pos_target.position.x = self.dp[count][0]
            self.pos_target.position.y = self.dp[count][1]  
            self.pos_target.position.z = self.dp[count][2] + 2.0
            
            self.pos_target.velocity.x = 0.0 
            self.pos_target.velocity.y = 0.0 #self.dp[count][4] 
            self.pos_target.velocity.z = 0.0 #self.dp[count][5] 
            self.pos_target.yaw = self.dp[count][6] 
            
            #print("PosTarget: ", self.pos_target.position.x, self.pos_target.position.y, self.pos_target.position.z, 
            #    self.pos_target.velocity.x, self.pos_target.velocity.y, self.pos_target.velocity.z,
            #    self.pos_target.yaw)
            
            self.pos_target_setpoint_pub.publish(self.pos_target)
            count+=1
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass


    def assertTrue(self, assertTrue, fail_msg):
        if not assertTrue:
            rospy.loginfo(fail_msg)
            
 
    '''
    def check_position_diff(self, pos, zy=True):
        """publish diff between target and desired position: in meters"""
            
        #desired = np.array((self.flattarget.position.x, self.flattarget.position.y, self.flattarget.position.z))
        if zy:
        desired = np.array((pos[0], pos[1], pos[2]))
        current = np.array((self.local_position.pose.position.x,
                        self.local_position.pose.position.y,
                        self.local_position.pose.position.z))
                        
        diff = np.linalg.norm(desired - current) 
                        
        rospy.loginfo(
            "trajectory x:{0:.4f}, y:{1:.4f}, z:{2:.4f}  |  current x:{3:.4f}, y:{4:.4f}, z:{5:.4f}  | diff:{6:.4f}".format(
                desired[0], desired[1], desired[2], current[0], current[1], current[2], diff))
                
        self.diff_pub.publish(diff)
    '''



    def check_position_diff(self, pos, zy=True):
        """publish diff between target and desired position: in meters"""
            
        #desired = np.array((self.flattarget.position.x, self.flattarget.position.y, self.flattarget.position.z))
        if zy:
            desired = np.array((pos[2]))
            current = np.array((self.local_position.pose.position.z))
        else:
            desired = np.array((pos[1]))
            current = np.array((self.local_position.pose.position.y))        
                        
        diff = abs(desired-current) #np.linalg.norm(desired - current) 
        
        if zy:                
            rospy.loginfo(
            "status: pub_pos | x:{0:.4f}, y:{1:.4f}, z:{2:.4f}  |  current x:{3:.4f}, y:{4:.4f}, z:{5:.4f}  | diff:{6:.4f}".format(
                pos[0], pos[1], pos[2], self.local_position.pose.position.x, self.local_position.pose.position.y, self.local_position.pose.position.z, diff))
        else:
            rospy.loginfo(
            "status: pub_vel | x:{0:.4f}, y:{1:.4f}, z:{2:.4f}  |  current x:{3:.4f}, y:{4:.4f}, z:{5:.4f}  | diff:{6:.4f}".format(
                pos[0], pos[1], pos[2], self.local_position.pose.position.x, self.local_position.pose.position.y, self.local_position.pose.position.z, diff))
                
        self.diff_pub.publish(diff)
        
        return diff
        

    def is_at_position(self, offset, x=0, y=0, z=0):
        """offset: meters"""
        rospy.logdebug(
            "current position | x:{0:.2f}, y:{1:.2f}, z:{2:.2f}".format(
                self.local_position.pose.position.x, self.local_position.pose.
                position.y, self.local_position.pose.position.z))

        desired = np.array((x, y, z))
        pos = np.array((self.local_position.pose.position.x,
                        self.local_position.pose.position.y,
                        self.local_position.pose.position.z))
        rospy.loginfo(
            "goto x:{0:.4f}, y:{1:.4f}, z:{2:.4f}  |  current x:{3:.4f}, y:{4:.4f}, z:{5:.4f}  | diff:{6:.4f}".format(
                desired[0], desired[1], desired[2], pos[0], pos[1], pos[2], np.linalg.norm(desired - pos) ))
        return np.linalg.norm(desired - pos) < offset


    def run_mission(self):
        rospy.loginfo("Start Mission")

        # arm the drone (optional)
        self.set_arm(True, 5)
        
        # turn on Offboard mode in QGroundControl manually
        #self.set_mode("OFFBOARD", 5)
        
        # fly to the initial height of 0.5m above ground (for physical flying in the lab)
        init_pos = (self.dp[0][0], self.dp[0][1], 0.5)
        self.goto_pos(init_pos[0], init_pos[1], init_pos[2])
        rospy.loginfo("Init position reached! {0:.6f} {1:.6f} {2:.6f}".format(init_pos[0], init_pos[1], init_pos[2]))
        
        # publish waypoints [0 ... n-2] without feedback for each waypoint
        self.send_pos_target()
        
        # attempt to converge to the 2nd last waypoint [n-1]
        self.goto_pos(self.dp[-2][0], self.dp[-2][1], self.dp[-2][2])
        
        # fly towards the last waypoint n with velocity
        self.send_vel()

        return

    
    # for geometric controller
    def send_flatsetpoint(self):
        from tf.transformations import quaternion_from_euler

        rate = rospy.Rate(10)  # Hz
        count = 0
        
        # turn on the geometric controller
        self.set_trajectory(True)
        
        while not rospy.is_shutdown() and count < self.dp.shape[0]:
            self.flattarget.header = Header()
            self.flattarget.header.frame_id = "flat"
            self.flattarget.type_mask = 4 #IGNORE_SNAP_JERK_ACC = 4
            
            self.flattarget.position.x = self.dp[count][0]
            self.flattarget.position.y = self.dp[count][1]  
            self.flattarget.position.z = self.dp[count][2] + 1.0

            print("FlatTarget: ", self.flattarget.position.x, self.flattarget.position.y, self.flattarget.position.z)

            self.flattarget.velocity.x = 0.0 #self.dp[count][3] 
            self.flattarget.velocity.y = 0.5 # self.dp[count][4]  #- 1.0
            self.flattarget.velocity.z = 0.5 # self.dp[count][5]  #- 2.0

            self.flattarget.header.stamp = rospy.Time.now()
            
            self.flatreference_pub.publish(self.flattarget)
            count += 1
            try:  
                rate.sleep()
            except rospy.ROSInterruptException:
                pass   

        self.set_trajectory(False)


    # arming function
    def set_arm(self, arm, timeout):
        """arm: True to arm or False to disarm, timeout(int): seconds"""
        rospy.loginfo("setting FCU arm: {0}".format(arm))
        old_arm = self.state.armed
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        arm_set = False
        for i in xrange(timeout * loop_freq):
            if self.state.armed == arm:
                arm_set = True
                rospy.loginfo("set arm success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_arming_srv(arm)
                    if not res.success:
                        rospy.logerr("failed to send arm command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                pass
                #self.fail(e)

        self.assertTrue(arm_set, (
            "failed to set arm | new arm: {0}, old arm: {1} | timeout(seconds): {2}".
            format(arm, old_arm, timeout)))
        return arm_set

    # set flight mode
    def set_mode(self, mode, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        rospy.loginfo("setting FCU mode: {0}".format(mode))
        old_mode = self.state.mode
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        mode_set = False
        for i in xrange(timeout * loop_freq):
            if self.state.mode == mode:
                mode_set = True
                rospy.loginfo("set mode success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_mode_srv(0, mode)  # 0 is custom mode
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                pass
                #self.fail(e)

        self.assertTrue(mode_set, (
            "failed to set mode | new mode: {0}, old mode: {1} | timeout(seconds): {2}".
            format(mode, old_mode, timeout)))

    # take off command
    def takeoff(self, timeout=10):
        """mode: PX4 mode string, timeout(int): seconds"""
        #rospy.loginfo("setting FCU mode: {0}".format(mode))
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        take_off = False
        for i in xrange(timeout * loop_freq):
            if not take_off:
                try:
                    self.set_takeoff_srv(altitude = 3.0)
                    take_off = True
                    rospy.loginfo("takeoff successful!")
                    break
                except rospy.ServiceException as e:
                    rospy.loginfo("takeoff call failed.. try again")
           
            try:
                rate.sleep()
            except rospy.ROSException as e:
                pass
                #self.fail(e)
        if take_off:
            rospy.loginfo("Waiting for take off 5 sec")
            rospy.sleep(5)
        return take_off

if __name__ == '__main__':
    rospy.init_node('offboard_mission_node')
    suction_mission = MavrosOffboardSuctionMission()
    suction_mission.run_mission()
    rospy.spin()





