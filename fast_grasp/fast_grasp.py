###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from kinematics import forward_kinematics
from scipy.spatial.transform import Rotation as R
import logging

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

from kortex_api.autogen.messages import Base_pb2

from kortex_api.Exceptions.KServerException import KServerException

import rospy
from geometry_msgs.msg import PoseStamped

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

def example_move_to_start_position(base, eval=False, iter=300):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    constrained_joint_angles = Base_pb2.ConstrainedJointAngles()

    actuator_count = base.GetActuatorCount().count
    angles = [0.0] * actuator_count

    # Actuator 4 at 90 degrees
    for joint_id in range(len(angles)):
        joint_angle = constrained_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = angles[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Reaching joint angles...")
    base.PlayJointTrajectory(constrained_joint_angles)

    print("Waiting for movement to finish ...")
    if eval:
        eval_performance(base, iter=iter)
        return True
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Joint angles reached")
    else:
        print("Timeout on action notification wait")
    return finished

def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    finished = True
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished

def example_send_joint_speeds(base, speeds):

    joint_speeds = Base_pb2.JointSpeeds()
    del joint_speeds.joint_speeds[:]
    i = 0
    for speed in speeds:
        joint_speed = joint_speeds.joint_speeds.add()
        joint_speed.joint_identifier = i 
        joint_speed.value = speed
        joint_speed.duration = 0
        i = i + 1
    
    base.SendJointSpeedsCommand(joint_speeds)
    # print ("Running", end="\r")
    return True

def example_send_gripper_command(base, value):
    # Create the GripperCommand we will send
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()

    # Close the gripper with position increments
    print("Performing gripper test in position...")
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = value
    base.SendGripperCommand(gripper_command)
    return True

def inverse_kinematics(base, T):
    ''' T: 4x4 homogeneous transformation matrix 
        return: joint angles in degrees'''
    q = np.zeros(6)
    p_x = T[0,3]
    p_y = T[1,3]
    p_z = T[2,3]

    RM = T[0:3, 0:3]
    thetas = R.from_matrix(RM).as_euler('xyz', degrees=True)

    # Object containing cartesian coordinates and Angle Guess
    input_IkData = Base_pb2.IKData()
    
    # Fill the IKData Object with the cartesian coordinates that need to be converted
    input_IkData.cartesian_pose.x = p_x
    input_IkData.cartesian_pose.y = p_y
    input_IkData.cartesian_pose.z = p_z
    input_IkData.cartesian_pose.theta_x = thetas[0]
    input_IkData.cartesian_pose.theta_y = thetas[1]
    input_IkData.cartesian_pose.theta_z = thetas[2]
    
    # Fill the IKData Object with the guessed joint angles
    for i in range(6):
        input_IkData.guess.joint_angles.add()
    try:
        print("Computing Inverse Kinematics using joint angles and pose...")
        computed_joint_angles = base.ComputeInverseKinematics(input_IkData)
    except KServerException as ex:
        print("Unable to compute inverse kinematics")
        print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
        print("Caught expected error: {}".format(ex))
        return q
    for i in range(6):
        q[i] = computed_joint_angles.joint_angles[i].value
    return q

def pid_tuning(base, pids):
    # pid tuning
    success = True
    success &= example_move_to_start_position(base)
    ref_position_final = np.array([0, 0, 0, 0, 0, 120])
    N = 200
    fdb_record = np.zeros([6, N])
    for i in range(N):
        print("Iteration: ", i, end="\r")
        base_feedback = base.GetMeasuredJointAngles()
        speeds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        fdb_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ref_position = ref_position_final
        for j in range(6):
            fdb_position[j] = base_feedback.joint_angles[j].value
            fdb_position[j] = warp_to_range(fdb_position[j])
            ref_position[j] = warp_to_range(ref_position[j])
            fdb_record[j, i] = fdb_position[j]
            speeds[j] = pids[j].control(ref_position[j], fdb_position[j])
        success &= example_send_joint_speeds(base, speeds=speeds)
        if not success:
            break
    for i in range(6):
        plt.plot(fdb_record[i, :])
        plt.legend(["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Joint 6"])
    plt.show()
    
def move_to_position_withpid(base, ref_position, pids, iter=10000):  
    ''' ref_position: in degrees, joint space'''
    success = True
    speeds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(6):
        ref_position[i] = warp_to_range(ref_position[i])
    success &= check_limit(ref_position)
    if not success:
        return False
    N = iter
    total_interition = iter
    tic = time.time()
    for i in range(N):
        # print("Iteration: ", i, end="\r")
    # while True:
        base_feedback = base.GetMeasuredJointAngles()
        fdb_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for j in range(6):
            fdb_position[j] = base_feedback.joint_angles[j].value
            fdb_position[j] = warp_to_range(fdb_position[j])
            speeds[j] = pids[j].control(ref_position[j], fdb_position[j])
            logging.info(f'Joint {j}: {fdb_position[j]}')
        success &= example_send_joint_speeds(base, speeds=speeds)
        error = np.linalg.norm(ref_position - fdb_position)
        print("Error: ", error, end="\r")
        if (error < 1 or not success) and iter == 10000:
            total_interition = i
            break
    toc = time.time()
    if total_interition == 0:
        total_interition = 1
    i_time = (toc - tic) / total_interition * 1000
    print("Average iteration time: ", i_time, "ms")
    if success:
        print("Position reached successfully with PID controller") 
    return success

def check_limit(ref_position):
    limit_position = np.array([153, 153, 149, 149, 144, 148])
    if np.all(abs(ref_position) < limit_position):
        return True
    else:
        print("Position out of limit")
        return False

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error_sum = 0
        self.error_last = 0

    def control(self, ref, fdb):
        error = ref - fdb
        self.error_sum = self.error_sum + error
        error_diff = error - self.error_last
        self.error_last = error
        control = self.Kp * error + self.Ki * self.error_sum + self.Kd * error_diff
        return control

def warp_to_range(value, min_value = -180, max_value = 180):
    while value > max_value:
        value = value - 360
    while value < min_value:
        value = value + 360
    return value

def warp_to_range_array(value, min_value = -180, max_value = 180):
    for i in range(len(value)):
        value[i] = warp_to_range(value[i], min_value, max_value)
    return value

def eval_performance(base, iter=300):
    logging.basicConfig(filename='joint_positions.log', filemode='w', level=logging.INFO, format='%(message)s')
    N = iter
    fdb_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(N):
        print("Iteration: ", i, end="\r")
        base_feedback = base.GetMeasuredJointAngles()
        for j in range(6):
            fdb_position[j] = base_feedback.joint_angles[j].value
            fdb_position[j] = warp_to_range(fdb_position[j])
            logging.info(f'Joint {j}: {fdb_position[j]}')

def main():
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        # Init
        base = BaseClient(router)
        Home_Position = np.array([0, -15, 75, 0, -60, 0])
        Zero_Position = np.array([0, 0, 0, 0, 0, 0])
        Ready_Position = np.array([6, -25, 72, -90, -80, -12])
        Final_Position = np.array([-34, -55, 68, -90, -55, -88])
        # log depend on the current time
        logging.basicConfig(filename='joint_positions.log', filemode='w', level=logging.INFO, format='%(message)s')
        
        pids = [PIDController(1.0, 0.0, 0.0) for _ in range(6)]
        pids[0] = PIDController(2.0, 0.0, 0.0)
        pids[1] = PIDController(1.0, 0.0, 0.0)
        pids[2] = PIDController(1.0, 0.0, 0.0)
        pids[3] = PIDController(1.5, 0.0, 0.0)
        pids[4] = PIDController(1.0, 0.0, 0.0)
        pids[5] = PIDController(4.0, 0.0, 0.0)

        success = True
        success &= move_to_position_withpid(base, Ready_Position, pids)
        success &= example_send_joint_speeds(base, [0, 0, 0, 0, 0, 0])        
        example_send_gripper_command(base, 0.0)
        time.sleep(2)
        
        rospy.init_node("PID_grasp")
        while not rospy.is_shutdown():
            grasp_pose_cart = rospy.wait_for_message("/grasp_pose", PoseStamped)
            if grasp_pose_cart is not None:
                break
        # 从位姿变换到矩阵
        grasp_pose_T = np.zeros((4, 4))
        grasp_pose_T[0:3, 0:3] = R.from_quat([grasp_pose_cart.pose.orientation.x, grasp_pose_cart.pose.orientation.y, grasp_pose_cart.pose.orientation.z, grasp_pose_cart.pose.orientation.w]).as_matrix()
        grasp_pose_T[0, 3] = grasp_pose_cart.pose.position.x
        grasp_pose_T[1, 3] = grasp_pose_cart.pose.position.y
        grasp_pose_T[2, 3] = grasp_pose_cart.pose.position.z + 0.03
        grasp_pose_T[3, 3] = 1
        grasp_pose = inverse_kinematics(base, grasp_pose_T)
        grasp_pose = warp_to_range_array(grasp_pose)
        grasp_pose_T_val = forward_kinematics(grasp_pose)
        if not np.allclose(grasp_pose_T, grasp_pose_T_val, atol=1e-2):
            print("Inverse kinematics failed")
            return 1
        print("Grasp position: ", grasp_pose)
        
        success &= move_to_position_withpid(base, grasp_pose, pids)
        example_send_gripper_command(base, 0.8)
        time.sleep(2)
        success &= move_to_position_withpid(base, Ready_Position, pids)
        success &= move_to_position_withpid(base, Final_Position, pids)
        example_send_gripper_command(base, 0.0)
        time.sleep(2)
        success &= move_to_position_withpid(base, Ready_Position, pids)

        return 0 if success else 1

if __name__ == "__main__":
    exit(main())
