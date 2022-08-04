#!/usr/bin/env python3
import numpy as np
import rospy
# import roslib; roslib.load_manifest('gazebo')
import tensorflow as tf
import time
from cv_bridge import CvBridge
import cv2
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.models import load_model

from std_srvs.srv import Empty
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState 
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg

from math import pi

import sys
import os

class RobotArm(object):

    def __init__(self):
        self.loop_rate = rospy.Rate(1)
        self.br = CvBridge()

        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("arm")
        self.gripper_group=moveit_commander.MoveGroupCommander("gripper")    
        self.grab_attempts=0
        print("Arm done")
        # Subscribers
        ##   Subscribe vision (image from camera)
        rospy.Subscriber("/camera_arm/rgb/image_raw", Image, self.visualServoying)
        print("subscribed")
        # ##   Subscribe robot state(current velocity, and arm state)
        # # rospy.Subscriber("/cmd_vel", Twist, self.velSub)
        # # rospy.Subscriber("/arm_controller/state", , self.velSub)

    def visualServoying(self, data):
        print("found image")
        image=self.br.imgmsg_to_cv2(data)/255.0
        cv2.imshow("Keypoints", image)
        cv2.waitKey(1)
        # r_mean=np.mean(data[:,:,0])
        # g_mean=np.mean(data[:,:,1])
        # b_mean=np.mean(data[:,:,2])
        detector = cv2.SimpleBlobDetector()
        keypoints = detector.detect(image)
        print("working")
        im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(0)


    def moveArmAngle(self, command):
        joint_angles=self.move_group.get_current_joint_values()
        joint_angles[0]=0.0
        joint_angles[1]=command[0]#1.2
        joint_angles[2]=command[1]#-0.4
        joint_angles[3]=command[2]#-0.8
        self.move_group.go(joint_angles,wait=True)
        self.move_group.stop()

    def moveArmPose(self, command):
        pose=self.move_group.get_current_pose().pose
        # pose_goal = geometry_msgs.msg.Pose()
        pose.position.x=command[0]
        pose.position.y=command[1]
        pose.position.z=command[2]
        self.move_group.set_pose_target(pose)
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success=self.move_group.go(wait=True)
        print("Outcome: ", success)
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        self.move_group.clear_pose_targets()
        
    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()

if __name__=='__main__':
    # coke_no=int(sys.argv[1])
    rospy.init_node("robotARM", anonymous=True)
    robot=RobotArm()
    robot.start()