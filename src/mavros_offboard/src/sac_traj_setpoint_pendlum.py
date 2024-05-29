#!/usr/bin/env python
import rospy
import math
import numpy as np
import time
from geometry_msgs.msg import PoseStamped, Quaternion, Point, Vector3, Twist, TwistStamped, WrenchStamped, AccelStamped
from mavros_msgs.msg import ParamValue
from mavros_msgs.msg import Altitude, ExtendedState, HomePosition, ParamValue, State, \
                            WaypointList, PositionTarget, AttitudeTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, ParamGet, ParamSet, SetMode, SetModeRequest, WaypointClear, \
                            WaypointPush, CommandTOL
from sensor_msgs.msg import NavSatFix, Imu
from controller_msgs.msg import FlatTarget
#from mavros_test_common import MavrosTestCommon
from pymavlink import mavutil
from six.moves import xrange
from std_msgs.msg import Header, Float64
from threading import Thread
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from multiprocessing import Value
from ctypes import c_int
from collections import deque
from std_srvs.srv import SetBool, SetBoolRequest
from os.path import expanduser

from datalogger import *
from scipy.spatial.transform import Rotation


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
        self.anglePoints = None
        self.vy = vy
        self.vNeeded = 2
        self.droneOrientation = 0
        try:
            ## Enter full path of the waypointn txt file here
            home = expanduser("~")
            filename = home + "/catkin_ws/src/mavros_offboard/src/velocety2_droneAngel39.txt"
            self.dp = np.loadtxt(filename, dtype='float', delimiter=' ', skiprows=2)
            rospy.loginfo("Read datapoint file for Approching.")
            #self.vy *= self.dp[-1][4] # correct the direction based on the last entry of the waypoint file
        except Exception as e:
            self.fail("Failed to read datapoint")

        try:
            ## Enter full path of the waypointn txt file here
            home = expanduser("~")
            filename = home + "/catkin_ws/src/mavros_offboard/src/shutdownControll-5000-plus.txt"
            self.anglePoints = np.loadtxt(filename, dtype='float', skiprows=2)
            rospy.loginfo("Read datapoint file for shutdown.")
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
        self.local_velocety = TwistStamped()
        self.mission_wp = WaypointList()
        self.state = State()
        self.offset = offset
        
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


    #
    # Helper methods
    #
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

        #print(data)

        temp = Rotation.from_quat([self.imu_data.orientation.x,self.imu_data.orientation.y,self.imu_data.orientation.z,self.imu_data.orientation.w])
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

    # ----------- HELPERS -----------
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

            quaternion = quaternion_from_euler(0.0, 0.0, 0.0) # roll, pitch, yaw angle
            self.pos.pose.orientation.x = quaternion[0]
            self.pos.pose.orientation.y = quaternion[1]
            self.pos.pose.orientation.z = quaternion[2]
            self.pos.pose.orientation.w = quaternion[3]

            self.pos.header.stamp = rospy.Time.now()
            self.pos_setpoint_pub.publish(self.pos)
            reached_pos = self.is_at_position(self.offset, x, y, z)

            if writeToDataLogger:
                self.saveDataToLogData(x,y,z)

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def saveDataToLogData(self,x,y,z):
            self.logData.appendStateData(rospy.Time.now().to_sec(),x,y,z \
                , self.local_position.pose.position.x,self.local_position.pose.position.y,self.local_position.pose.position.z \
                ,self.local_velocety.twist.linear.x,self.local_velocety.twist.linear.y, self.local_velocety.twist.linear.z)

    def returnDifference(self, pos):
        diff = ((self.local_position.pose.position.x-pos[0])**2+(self.local_position.pose.position.y-pos[1])**2+(self.local_position.pose.position.z-pos[2])**2)**0.5
        return diff

    def send_pos_raw(self, xOffset = 0, yOffset = 0, zOffset = 0 ):

        rate = rospy.Rate(20)  # Hz
        vel = 0.1
        count = 0
        attemts = 0
        self.pos_target = PositionTarget()
        
        intitalDistanceToNext = self.returnDifference([self.dp[count][0]+xOffset-self.dp[0][0],self.dp[count][1]+yOffset-self.dp[0][1],self.dp[count][2]+zOffset-self.dp[0][2]])
        xLast = self.dp[count][0]
        yLast = self.dp[count][1]
        zLast = self.dp[count][2]
        
        while not rospy.is_shutdown() and count < self.dp.shape[0]:
            attemts += 1 
            self.pos_target.header = Header()
            self.pos_target.header.frame_id = "datapoint"
            self.pos_target.header.stamp = rospy.Time.now()
            self.pos_target.type_mask = PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ + \
                                        PositionTarget.FORCE + PositionTarget.IGNORE_YAW_RATE # + \
                                        #PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ 
            self.pos_target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

            xNext = self.dp[count][0]+xOffset-self.dp[0][0]
            yNext = self.dp[count][1]+yOffset-self.dp[0][1]  
            zNext = self.dp[count][2]+zOffset-self.dp[0][2] 

            self.pos_target.position.x = xNext
            self.pos_target.position.y = yNext  
            self.pos_target.position.z = zNext
            
            self.pos_target.velocity.x = 0.0 
            self.pos_target.velocity.y = -self.dp[count][4] * vel
            self.pos_target.velocity.z = -self.dp[count][5] * vel
            self.pos_target.yaw = self.dp[count][6] 
            
            #print("PosTarget: ", self.pos_target.position.x, self.pos_target.position.y, self.pos_target.position.z, 
            #    self.pos_target.velocity.x, self.pos_target.velocity.y, self.pos_target.velocity.z,
            #    self.pos_target.yaw)
            
            self.pos_target_setpoint_pub.publish(self.pos_target)

            print(self.returnDifference([xNext,yNext,zNext])-intitalDistanceToNext, count)

            self.saveDataToLogData(xNext,yNext,zNext)

            if (self.returnDifference([xLast,yLast,zLast]) > intitalDistanceToNext-self.offset/2) or attemts > 50:
                count+=1
                print(attemts)
                attemts = 0
                if count<self.dp.shape[0]:
                    intitalDistanceToNext = self.returnDifference([self.dp[count][0]+xOffset-self.dp[0][0],self.dp[count][1]+yOffset-self.dp[0][1],self.dp[count][2]+zOffset-self.dp[0][2]])
                xLast = xNext
                yLast = yNext
                zLast = zNext
                print("inital: ", intitalDistanceToNext)
            
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        
    def is_at_position(self, offset, x=0, y=0, z=0, printOut = True):
        """offset: meters"""
        rospy.logdebug(
            "current position | x:{0:.2f}, y:{1:.2f}, z:{2:.2f}".format(
                self.local_position.pose.position.x, self.local_position.pose.
                position.y, self.local_position.pose.position.z))

        desired = np.array((x, y, z))
        pos = np.array((self.local_position.pose.position.x,
                        self.local_position.pose.position.y,
                        self.local_position.pose.position.z))

        if printOut:
            rospy.loginfo(
                "goto x:{0:.4f}, y:{1:.4f}, z:{2:.4f}  |  current x:{3:.4f}, y:{4:.4f}, z:{5:.4f}  | diff:{6:.4f}".format(
                    desired[0], desired[1], desired[2], pos[0], pos[1], pos[2], np.linalg.norm(desired - pos) ))
        return np.linalg.norm(desired - pos) < offset

    # ----------- FLIGHT PATH METHODS -----------

    def swingPendelum(self, centerX, centerY, centerZ):

        freqenzy = 20
        rate = rospy.Rate(freqenzy)  # Hz
        waiting = False

        amplitude = 1

        PointInfront = [centerX, centerY+amplitude, centerZ]
        PointBehind = [centerX, centerY-amplitude, centerZ]
        LastPoint = PointInfront

        intialDifCenter = self.returnDifference([centerX, centerY, centerZ])

        direction = -1
        count = 0
        attempts = 0
        secondsWaiting = 2

        while not rospy.is_shutdown() and count < 3:
            self.pos_target.header = Header()
            self.pos_target.header.frame_id = "velocity_pos"
            self.pos_target.header.stamp = rospy.Time.now()
            self.pos_target.type_mask = PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ + \
                                        PositionTarget.FORCE + PositionTarget.IGNORE_YAW_RATE +  \
                                        PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ
            self.pos_target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

            self.pos_target.position.x = centerX
            self.pos_target.position.y = centerY
            self.pos_target.position.z = centerZ

            self.pos_target.velocity.x = 0
            self.pos_target.velocity.y = 1.7 * direction
            self.pos_target.velocity.z = 0
            self.pos_target.yaw = 0

            self.pos_target.type_mask = 0b0000101111000000 #0b0 000 000 000 000 000 ignor yarn acc vel pos

            self.pos_target_setpoint_pub.publish(self.pos_target)

            if self.returnDifference(LastPoint) > 2*amplitude:
                direction *= -1
                count += 1
                attempts = 0
                #waiting= True
                if LastPoint == PointInfront:
                    LastPoint = PointBehind
                else:
                    LastPoint = PointInfront

            ## make if call here counting the number of attemtps if more than 3 seconts before increase counter -> connection break loop!

            if attempts > freqenzy*secondsWaiting: # assume conection is made -> can not reach target 
                rospy.loginfo("Connection made! continue ...")
                break

            attempts += 1 

            self.saveDataToLogData(centerX, centerY, centerZ)

            print(count,"- distance to next one: ", self.returnDifference([centerX, centerY, centerZ]), "distance to last one: ", self.returnDifference(LastPoint))

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass     

    def navigate_to_starting_position(self, rate, initX, initY, initZ, last_req):
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'

        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True

        while(not rospy.is_shutdown()):
            if(self.state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                    rospy.loginfo("OFFBOARD enabled")
                
                last_req = rospy.Time.now()
            else:
                if(not self.state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                    if(self.arming_client.call(arm_cmd).success == True):
                        rospy.loginfo("Vehicle armed")
                
                    last_req = rospy.Time.now()

            self.pos_setpoint_pub.publish(self.pos)

            if self.is_at_position(self.offset,initX, initY, initZ, printOut=False):
                rospy.loginfo("INITIAL POSITION REACHED")
                break

            
            rate.sleep()
        
        return last_req

    # using datapoint file and iterate over the datapoint for pitch angle
    def auto_send_landing_pos_att(self):
        rate = rospy.Rate(20)  # Hz
        count = 0
        loop_file = True
        
        while not rospy.is_shutdown():  
            if True:
                if loop_file:
                     count += 1
                     if count >= self.anglePoints.shape[0]:
                        count = -1
                        loop_file = False
                self.att_setpoint_pub.publish(self.att_raw_msg(count))
            else:
                self.pos_target_setpoint_pub.publish(self.pos_raw_msg(None))
            self.saveDataToLogData(0,0,0)
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
                
        rospy.loginfo("Finish sending setpoints!") 
    
    # ----------- HOVER -----------
    # got to position and stay there for defined amount of time
    def hoverAtPos(self, x, y, z, time):

        #go to pos
        self.goto_pos(x, y, z)

        pose = PoseStamped()
        frequence = 20
        waitingTime = time * frequence
        rate = rospy.Rate(frequence)  # Hz

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        # sent point until time is over
        for i in range(waitingTime):
            if(rospy.is_shutdown()):
                break

            self.pos_setpoint_pub.publish(pose)
            self.saveDataToLogData(x,y,z)
            rate.sleep()

        rospy.loginfo("Waiting Over")

    # ----------- FLIGHT PATH METHODS -----------
    def run_full_mission(self, xTarget = 0, yTarget = -2, zTarget = 1):

        hightOverBranch = 0.8
        ropeLengt = 1.6

        # Setpoint publishing MUST be faster than 2Hz
        rate = rospy.Rate(20)

        # Wait for Flight Controller connection
        while(not rospy.is_shutdown() and not self.state.connected):
            rate.sleep()

        initX = self.local_position.pose.position.x
        initY = self.local_position.pose.position.y
        initZ = self.local_position.pose.position.z + 2 ##take off 2m 

        self.pos.pose.position.x = initX
        self.pos.pose.position.y = initY
        self.pos.pose.position.z = initZ

        # Send a few setpoints before starting
        for i in range(100):   
            if(rospy.is_shutdown()):
                break

            self.pos_setpoint_pub.publish(self.pos)
            rate.sleep()

        last_req = self.navigate_to_starting_position(rate, initX, initY, initZ, last_req=rospy.Time.now())

        ## go to start
        xOffset = xTarget + self.dp[0][0]-self.dp[-1][0]
        yOffset = yTarget + self.dp[0][1]-self.dp[-1][1]+0.5 ##stop in front of the branch do not need this any more?
        zOffset = zTarget + self.dp[0][2]-self.dp[-1][2]+hightOverBranch ## offset over the branch 
        initX = xOffset
        initY = yOffset
        initZ = zOffset

        rospy.loginfo("---- HOVER @ STARTING ----")
        self.hoverAtPos(initX, initY, initZ, 5)

        rospy.loginfo("Go To Pos for Step 1")
        self.goto_pos(xOffset, yOffset, zOffset, writeToDataLogger=False)
        self.hoverAtPos(xOffset, yOffset, zOffset, 5)

        rospy.loginfo("Step 1")
        self.send_pos_raw(xOffset, yOffset, zOffset)

        rospy.loginfo("Pendelum Swinging")
        #self.swingPendelumViaPosition(xTarget, yTarget, zTarget+hightOverBranch)
        self.swingPendelum(xTarget, yTarget, zTarget+hightOverBranch)
        self.goto_pos(xTarget, yTarget, zTarget+hightOverBranch)

        rospy.loginfo("Step 2")
        self.hoverAtPos(xTarget, yTarget, zTarget+hightOverBranch, 4)

        rospy.loginfo("Go To Pos for Step 3")
        self.goto_pos(xTarget, -0.5, 1.45) # yTarget-ropeLengt/3, zTarget)
        self.hoverAtPos(xTarget, -0.5, 1.45, 4)

        rospy.loginfo("Step 3")
        self.auto_send_landing_pos_att() 

        # go to original pos
        rospy.loginfo("---- LAND ----")
        self.goto_pos(initX,initY,initZ, writeToDataLogger=False)
        land_set_mode = SetModeRequest()
        land_set_mode.custom_mode = 'AUTO.LAND'

        last_req = rospy.Time.now()

        while(not rospy.is_shutdown() and self.state.armed):
            if(self.state.mode != "AUTO.LAND" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                if(self.set_mode_client.call(land_set_mode).mode_sent == True):
                    rospy.loginfo("AUTO.LAND enabled")

                last_req = rospy.Time.now()
            self.pos_setpoint_pub.publish(self.pos)
            rate.sleep()

    def att_raw_msg(self, dp_count):
        att_target = AttitudeTarget()
        att_target.header = Header()
        att_target.header.frame_id = "landing_att"
        att_target.header.stamp = rospy.Time.now()
        #att_target.type_mask = AttitudeTarget.IGNORE_ROLL_RATE + AttitudeTarget.IGNORE_ROLL_RATE + AttitudeTarget.IGNORE_ROLL_RATE
        att_target.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        
        if dp_count is None:
            vector3 = Vector3(x=self.pitch_input , y=0.0 , z=0.0)
            att_target.thrust = self.thrust_input
        else:
            if dp_count > -1:
                vector3 = Vector3(x=self.anglePoints[dp_count] , y=0.0 , z=0.0)
                att_target.thrust = 0.35 ###0.3
            else:
                vector3 = Vector3(x=self.anglePoints[-1] , y=0.0 , z=0.0)
                att_target.thrust = 0.0
            rospy.loginfo("current pitch rate from file = {0:.6f}".format(self.anglePoints[dp_count]))


        att_target.body_rate = vector3

        return att_target


if __name__ == '__main__':
    rospy.init_node('offboard_mission_node')
    suction_mission = MavrosOffboardSuctionMission()
    suction_mission.run_full_mission(xTarget = 0.65, yTarget = 0.45, zTarget = 1.4)

    suction_mission.logData.saveAll()
    suction_mission.logData.plotFigure()
    rospy.loginfo("huhu")
