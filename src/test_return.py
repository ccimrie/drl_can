#!/usr/bin/env python3
import numpy as np
import rospy
# import roslib; roslib.load_manifest('gazebo')
# import tensorflow as tf
import time
# from cv_bridge import CvBridge
# import cv2
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import load_model

from std_srvs.srv import Empty
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int8
from sensor_msgs.msg import JointState 
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *

from math import pi

import sys
import os

class RobotArm(object):

    def __init__(self):
        self.loop_rate = rospy.Rate(1)
        self.arm=0
        self.move_group = moveit_commander.MoveGroupCommander("arm")
        self.gripper_group=moveit_commander.MoveGroupCommander("gripper")    

        ## For aligning with wall
        self.scan=rospy.Subscriber('/dummy', Int8, self.align)
        self.e_left_prev=0.0
        self.e_right_prev=0.0

        self.vel_pub=rospy.Publisher("/cmd_vel", Twist,queue_size=1)

        # Tell the action client that we want to spin a thread by default
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.init_pose=rospy.Publisher("/initialpose",geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=1)
        rospy.loginfo("Wait for the action server to come up")
        self.setInitialPose([0.0,0.0,0],[0.0,1.0])
        rospy.Subscriber("/arm_motion", Int8, self.armUpdate)

        # Allow up to 5 seconds for the action server to come up
        self.move_base.wait_for_server(rospy.Duration(5))

    def armUpdate(self, data):
        print("Arm change")
        print(data)
        arm_temp=data.data
        self.arm=arm_temp
        self.returnHome()
        print("part 1 done")
        self.moveArmAngle([0.75,-0.75,0.0])
        print("part 2 done")
        self.gripperOpen()
        print("part 3 done")
        self.moveArmHome()
        self.returnSearch()

    def moveArmAngle(self, command):
        joint_angles=self.move_group.get_current_joint_values()
        joint_angles[0]=0.0
        joint_angles[1]=command[0]#1.2
        joint_angles[2]=command[1]#-0.4
        joint_angles[3]=command[2]#-0.8
        self.move_group.go(joint_angles,wait=True)
        self.move_group.stop()

    # def moveArmPose(self, command):
    #     pose=self.move_group.get_current_pose().pose
    #     # pose_goal = geometry_msgs.msg.Pose()
    #     pose.position.x=pose.position.x+command[0]
    #     pose.position.y=pose.position.y+command[1]
    #     pose.position.z=pose.position.z+command[2]
    #     self.move_group.set_pose_target(pose)
    #     # `go()` returns a boolean indicating whether the planning and execution was successful.
    #     success=self.move_group.go(wait=True)
    #     print("Outcome: ", success)
    #     # Calling `stop()` ensures that there is no residual movement
    #     self.move_group.stop()
    #     # It is always good to clear your targets after planning with poses.
    #     # Note: there is no equivalent function for clear_joint_value_targets().
    #     self.move_group.clear_pose_targets()
    
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

    def returnSearch(self):
        # Send a goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = geometry_msgs.msg.Pose(geometry_msgs.msg.Point(3.5, 3.5, 0.000),
                                         geometry_msgs.msg.Quaternion(0.0, 0, -3*np.pi/4.0, 1.0))

        # Start moving
        self.move_base.send_goal(goal)
        # Allow TurtleBot up to 60 seconds to complete task
        # success = self.move_base.wait_for_result(rospy.Duration(60)) 
        state = self.move_base.get_state()
        print(state)
        while state==GoalStatus.ACTIVE or state==GoalStatus.PENDING:
            state=self.move_base.get_state()
            print(state)
        print("Searching")
        
        return  

    def returnHome(self):
        # Send a goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = geometry_msgs.msg.Pose(geometry_msgs.msg.Point(4.5, 3.25, 0.000),
                                         geometry_msgs.msg.Quaternion(0.0, 0, np.pi/2, 1.0))

        # Start moving
        self.move_base.send_goal(goal)
        # Allow TurtleBot up to 60 seconds to complete task
        # success = self.move_base.wait_for_result(rospy.Duration(60)) 
        state = self.move_base.get_state()
        print(state)
        while state==GoalStatus.ACTIVE or state==GoalStatus.PENDING:
            state=self.move_base.get_state()
            print(state)
        print("Home")
        
        self.scan=rospy.Subscriber('/scan', LaserScan, self.align)
        self.aligned=0
        while not self.aligned:
            time.sleep(0.5)
        self.scan.unregister()
        return    

    def align(self, data):
        vals_left=np.array(data.ranges[0:7])
        # print(np.array(vals_left))
        vals_right=np.array(data.ranges[-1:-6:-1])

        goal=0.15

        e_left=vals_left-goal
        e_right=vals_right-goal

        Kp=0.5
        Kd=0.1
        Ki=0.0

        ml=np.sum(Kp*e_left+Kd*(e_left-self.e_left_prev))
        mr=np.sum(Kp*e_right+Kd*(e_right-self.e_right_prev))

        if ml>0.15:
            ml=0.15
        elif ml<-0.15:
            ml=-0.15
        if mr>0.15:
            mr=0.15
        elif mr<-0.15:
            mr=-0.15

        self.e_left_prev=e_left
        self.e_right_prev=e_right

        # float speed_wish_right = (cmd_vel.angle*WHEEL_DIST)/2 + cmd_vel.speed;
        # float speed_wish_left = cmd_vel.speed*2-speed_wish_right;

        vel=Twist()
        vel.linear.x=(ml+mr)/2.0
        # ml_actual=ml+noise
        # vel.linear.x=(ml_actual+mr)/2.0
        vel.angular.z=(2*(mr-vel.linear.x))/0.306
        self.vel_pub.publish(vel)

        e_bound=0.1
        if (np.sum(e_left)<e_bound and np.sum(e_right)<e_bound):
            self.aligned=1
            vel.linear.x=0.0
            vel.angular.z=0.0
            self.vel_pub.publish(vel)

    def setInitialPose(self,pos, orientation):
        pose = geometry_msgs.msg.PoseWithCovarianceStamped()
        pose.header.frame_id = "map"
        pose.pose.pose.position.x=pos[0]
        pose.pose.pose.position.y=pos[1]
        pose.pose.pose.position.z=pos[2]
        pose.pose.covariance=[0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]
        pose.pose.pose.orientation.z=orientation[0]
        pose.pose.pose.orientation.w=orientation[1]
        # rospy.loginfo(pose)
        # rospy.spin()
        # rospy.loginfo(pose)
        self.loop_rate.sleep()
        self.init_pose.publish(pose)

    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()

if __name__=='__main__':
    # coke_no=int(sys.argv[1])
    rospy.init_node("robotARM", anonymous=True)
    robot=RobotArm()
    robot.start()