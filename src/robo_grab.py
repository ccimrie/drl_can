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
from std_msgs.msg import Int8
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
        self.arm=0
        self.timer=0
        print("Arm done")
        # Subscribers
        ##   Subscribe vision (image from camera)
        rospy.Subscriber("/camera_arm/rgb/image_raw", Image, self.visualServoying)
        rospy.Subscriber("/arm_motion", Int8, self.armUpdate)
        print("subscribed")
        # ##   Subscribe robot state(current velocity, and arm state)
        # # rospy.Subscriber("/cmd_vel", Twist, self.velSub)
        # # rospy.Subscriber("/arm_controller/state", , self.velSub)
        self.vel_pub=rospy.Publisher("/cmd_vel", Twist,queue_size=1)
        self.arm_update_pub=rospy.Publisher("/arm_motion",Int8,queue_size=1)

        ## Set if demoing this node only
        ## Initiate grab
        self.gripperOpen()
        self.moveArmAngle([1.1,-0.1,-1.0])
        self.arm=1

    def armUpdate(self, data):
        print("Arm change")
        print(data)
        arm_temp=data.data
        if arm_temp==1:
            twist_vel=Twist()
            twist_vel.linear.x=0.0
            twist_vel.angular.z=0.0
            self.vel_pub.publish(twist_vel)
            self.moveArmHome()
            self.gripperOpen()
            ## Initiate grab
            self.moveArmAngle([1.1,-0.1,-1.0])
        self.arm=arm_temp


    def visualServoying(self, data):
        if self.arm==0:
            # print("No action")
            return
        elif self.timer>700:
            self.arm_update_pub.publish(0)
            self.moveArmHome()
            self.timer=0
            return
        else: 
            # print("found image")
            self.timer=self.timer+1
            if self.timer%100==0:
                print(self.timer)

        twist=Twist()
        # twist.linear.x=0.0
        # twist.angular.z=0.0
        # self.vel_pub.publish(twist)
        # print("Stopping")


        image=self.br.imgmsg_to_cv2(data)[0:450,:]#,"bgr8")#/255.0
        centre_x=np.shape(image)[1]/2
        # centre_y=np.shape(image)[0]/2
        centre_y=int(0.99*np.shape(image)[0])

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image=np.float32(image)
        # image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image=cv2.bitwise_not(image)
        # cv2.imshow("Keypoints", image)
        # cv2.waitKey(1000)
        # r_mean=np.mean(data[:,:,0])
        # g_mean=np.mean(data[:,:,1])
        # b_mean=np.mean(data[:,:,2])
        # params = cv2.SimpleBlobDetector_Params()
        # detector = cv2.SimpleBlobDetector_create(params)

        # # detector.empty() # <- now works
        # keypoints = detector.detect(image) # <- now works
        # print(keypoints)
        # print("working")
        # im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # # Show keypoints
        # cv2.imshow("Keypoints", im_with_keypoints)
        # cv2.waitKey(10)

        image2=cv2.Canny(image,0,50)

        kernel = np.ones((1,20), np.uint8)  # note this is a horizontal kernel
        d_im = cv2.dilate(image2, kernel, iterations=20)
        image2 = cv2.erode(d_im, kernel, iterations=5)         

        # # get the (largest) contour
        contours = cv2.findContours(image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)

        # # draw white filled contour on black background
        result = np.zeros_like(image2)
        cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)

        M = cv2.moments(result)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # cY=int(0.95*np.shape(image)[0])
        # put text and highlight the center
        cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)
        cv2.putText(image, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        bound=10

        ## check horizontal alignment first
        if cX<centre_x-bound:
            ## turn left
            # twist.linear.x=0
            twist.angular.z=0.02
            # self.vel_pub.publish(twist)
            # print("TURNING LEFT")
            # time.sleep(0.1)
        elif cX>centre_x+bound:
            ## turn right
            # twist.linear.x=0
            twist.angular.z=-0.02
            # self.vel_pub.publish(twist)
            # print("TURNING RIGHT")
        else:
            # print("NO turning")
            twist.angular.z=0.0
            # time.sleep(0.1)
        ## check vertical alignment if centred
        if cY>centre_y+bound:
            ## pull back
            # twist.angular.z=0.0
            twist.linear.x=-0.005
            # self.moveArmPose([0,0,-0.001])
            # print("BACKWARDS")
            # time.sleep(0.1)
        elif cY<centre_y-bound:
            # self.moveArmPose([0,0,0.001])
            # twist.angular.z=0.0
            twist.linear.x=0.005
            # print("FORWARDS")
            # time.sleep(0.1)
        else:
            twist.linear.x=0
        self.vel_pub.publish(twist)
        if twist.linear.x==0 and twist.angular.z==0:
            self.gripperClose()
            self.moveArmHome()
            self.arm_update_pub.publish(0)
            self.arm=0
        # display the image
        # cv2.imshow("blobs centre", result)

        cv2.imshow("image", image)
        # cv2.imshow("blobs", result)
        cv2.waitKey(10)

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
        pose.position.x=pose.position.x+command[0]
        pose.position.y=pose.position.y+command[1]
        pose.position.z=pose.position.z+command[2]
        self.move_group.set_pose_target(pose)
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success=self.move_group.go(wait=True)
        print("Outcome: ", success)
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        self.move_group.clear_pose_targets()
    
    def moveArmHome(self):
        joint_angles=self.move_group.get_current_joint_values()
        joint_angles[0]=0.0
        joint_angles[1]=0.0
        joint_angles[2]=0.0
        joint_angles[3]=0.0
        self.move_group.go(joint_angles,wait=True)
        self.move_group.stop()

    def gripperOpen(self):
        joint_gripper=self.gripper_group.get_current_joint_values()
        joint_gripper[0]=0.018
        joint_gripper[1]=0.018
        self.gripper_group.go(joint_gripper, wait=True)
        self.gripper_group.stop()

    def gripperClose(self):
        joint_gripper=self.gripper_group.get_current_joint_values()
        joint_gripper[0]=-0.01
        joint_gripper[1]=-0.01
        self.gripper_group.go(joint_gripper, wait=True)
        self.gripper_group.stop()


    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()

if __name__=='__main__':
    # coke_no=int(sys.argv[1])
    rospy.init_node("robotARM", anonymous=True)
    robot=RobotArm()
    robot.start()