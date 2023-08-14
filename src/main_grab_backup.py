#!/usr/bin/env python3
import numpy as np
import rospy
# import roslib; roslib.load_manifest('gazebo')
import time
from cv_bridge import CvBridge
import cv2

from A2C import A2C

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
from darknet_ros_msgs.msg import ObjectCount
from darknet_ros_msgs.msg import BoundingBoxes
import geometry_msgs.msg
# import moveit_commander
# import moveit_msgs.msg

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *

from math import pi

import sys
import os

## Custom classes
from gazebo_utils import GazeboUtils as GU
print("HERE1")
from turtlebot_movement import TurtlebotMovement as NAV
print("HERE2")
from robot_arm import RobotArm
print("HERE3")

class RobotRL(object):

    def __init__(self, coke_no, light_no):
        self.loop_rate = rospy.Rate(1)
        self.br = CvBridge()

        ## Number of coke cans; setting up naming convention
        self.coke_no=coke_no
        self.coke_list=[]
        for i in np.arange(1,coke_no+1):
            coke="coke_"+str(i)
            self.coke_list.append(coke)

        ## Number of light fixtures; setting up naming convention
        self.light_no=light_no
        self.light_list=[]
        for i in np.arange(1,light_no+1):
            light="light_"+str(i)
            self.light_list.append(light)

        # self.state_size=20
        self.state_size=4
        width=640
        height=480

        self.width_state_size=int(width/self.state_size)
        self.height_state_size=int(height/self.state_size)

        self.state_size=self.width_state_size*self.height_state_size

        self.coke_box_state=np.zeros([self.height_state_size, self.width_state_size])
        self.light_box_state=np.zeros([self.height_state_size, self.width_state_size])
        print("Part 1 complete")
        # Network
        ## Check if a model is already created
        #path=os.environ["MODEL_PATH"]
        print("Creating RL agent")
        self.rl_agent=A2C(2, 2*self.width_state_size*self.height_state_size)
        print("Success!")
        self.episode=0
        self.learn_eps=500
        self.total_eps=0

        eps_no=1
        self.max_eps=self.learn_eps*eps_no

        self.f=open('rewards.txt', 'w+')

        ## For noetic/python3
        self.optimizer=keras.optimizers.Adam(learning_rate=0.01)
        ## For melodic/python2
        ## self.optimizer=keras.optimizers.Adam(lr=0.01)
        self.arm_trigger=0
        self.total_episode_reward=0

        print("Importing classes")
        ## Call custom classes
        self.gu=GU()         # Gazebo utility functions
        print("Utils success!")
        self.nav=NAV()       # Move base and navigation
        print("Navigation success!")
        self.arm=RobotArm()  # Control of arm
        print("Arm success!")

        ## Publishers
        self.vel_pub=rospy.Publisher("/cmd_vel", Twist,queue_size=1)
        self.arm_update_pub=rospy.Publisher("/arm_motion",Int8,queue_size=1)
        print("Publishers setup")
        ## Subscribers
        # rospy.Subscriber("/camera/rgb/image_raw", Image, self.imageSub)
        rospy.Subscriber("/arm_motion", Int8, self.armUpdate)

        ## Need to get number of objects detected, if >0 see what latest bounding boxes are
        rospy.Subscriber("/darknet_ros/found_object", ObjectCount, self.imageSub, queue_size=1)
        rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.storeBoxes, queue_size=1)
        print("Subscribers setup")

    def armUpdate(self, data):
        arm_temp=data.data
        print(self.arm_trigger)
        if arm_temp==0:
            self.moveArmHome()
            print("Arm change: check for reward")
            pos_coke=self.gms_client("coke_1","link").pose.position
            ## Assume last reward step was for grab attempt
            if pos_coke.z>0.1:
                self.rewards_history[-1]=self.rewards_history[-1]+100
                twist=Twist()
                twist.linear.x=0.0
                twist.angular.z=0.0
                self.vel_pub.publish(twist)
                self.imageSubLearn()

                ## Save RL network
                ## TODO

                self.gu.resetWorld()
            else:
                self.rewards_history[-1]=self.rewards_history[-1]-5
                self.moveArmHome()
        self.arm_trigger=arm_temp

    def storeBoxes(self,data):
        ##self.coke_box_state
        width=640.0
        height=480.0

        self.coke_box_state[:]=0
        self.light_box_state[:]=0

        for box in data.bounding_boxes:
            # print("Doing a box!", self.width_state_size, self.height_state_size)
            xmin=int((box.xmin/width)*(self.width_state_size-1))
            xmax=int((box.xmax/width)*(self.width_state_size-1))
            ymin=int((box.ymin/height)*(self.height_state_size-1))
            ymax=int((box.ymax/height)*(self.height_state_size-1))
            val=0
            if box.Class=="coke":
                val=1
                for x in np.arange(xmin, xmax+1):
                    for y in np.arange(ymin, ymax+1):
                        self.coke_box_state[y,x]=1
            else:
                val=2
                for x in np.arange(xmin, xmax+1):
                    for y in np.arange(ymin, ymax+1):
                        self.light_box_state[y,x]=1

            #cv2.imshow('coke', self.coke_box_state)
            #cv2.waitKey(1)
            #for x in np.arange(xmin, xmax+1):
            #    for y in np.arange(ymin, ymax+1):
            #        if self.box_state[y,x]==0:
            #            self.box_state[y,x]=val
            #        elif self.box_state[y,x]!=val:
            #            self.box_state[y,x]=3

    def imageSub(self, data):
        if self.arm_trigger==1:
            return

        if self.episode%100==0:
            print(self.episode)

        # Get state
        image_coke=self.coke_box_state.flatten()
        image_light=self.light_box_state.flatten()      
        image=np.append(image_coke,image_light)

        ## Send state to the networks and get actions
        act=self.rl_agent.step(image)

        act[0]=np.clip(act[0],-1,1)
        act[1]=np.clip(act[1],-1,1)

        ## Check if a grab or move action:
        # p=np.random.random()
        twist_vel=Twist()
        
        grab_thresh=0.05
        if abs(act[0])>grab_thresh or abs(act[1])>grab_thresh:
            twist_vel.linear.x=act[0]*0.15
            twist_vel.angular.z=act[1]*0.15
            self.vel_pub.publish(twist_vel)
        else:
            twist_vel.linear.x=0.0
            twist_vel.angular.z=0.0
            self.vel_pub.publish(twist_vel)

            pos_robot=self.gms_client("robot", "").pose.position
            reward=0

            ideal_dist=0.4

            for i in np.arange(self.coke_no):
                pos_coke=self.gms_client("coke_"+str(i),"link").pose.position
                dist=(pos_coke.x-pos_robot.x)**2+(pos_coke.y-pos_robot.y)**2
                reward_temp=1.0/((np.sqrt(dist)-0.4)**2+1e-1)

            self.rewards_history.append(reward)
            self.episode=self.episode+1
            self.arm_update_pub.publish(1)
            return

        rospy.sleep(0.1)
    
        twist_vel.linear.x=0.0
        twist_vel.angular.z=0.0
        self.vel_pub.publish(twist_vel)

        # Get reward
        # Reward is distance between coke and robot optimal is 0.2 distance
        ## Assumption: always at least one coke can
        pos_coke=self.gms_client("coke_1","").pose.position
        pos_robot=self.gms_client("robot", "").pose.position

        dist=(pos_coke.x-pos_robot.x)**2+(pos_coke.y-pos_robot.y)**2

        for i in self.coke_list[1:]:
            pos_coke_temp=self.gms_client(i,"").pose.position
            dist_temp=(pos_coke_temp.x-pos_robot.x)**2+(pos_coke_temp.y-pos_robot.y)**2
            if dist_temp<dist:
                dist=dist_temp
                pos_coke=pos_coke_temp
        
        reward_observe=float(np.sum(image_coke>0))/self.state_size
        if reward_observe==0:
            reward=-1
        else:
            reward_dist=1.0/((np.sqrt(dist)-0.35)**2+1e-2)
            # reward=reward_dist*reward_observe
            reward=reward_dist
        
        self.rl_agent.recordReward(reward)
        self.total_episode_reward=self.total_episode_reward+reward

        ## Add 'state decay'
        memory_thresh=0.2
        self.coke_box_state=self.coke_box_state*0.9
        self.light_box_state=self.light_box_state*0.9
        self.coke_box_state[self.coke_box_state<memory_thresh]=0
        self.light_box_state[self.light_box_state<memory_thresh]=0

        ## After fixed episode length learn
        ## (should also learn if robot completes dropping off?)
        if self.episode>self.learn_eps:
            twist=Twist()
            twist_vel.linear.x=0.0
            twist_vel.angular.z=0.0
            self.vel_pub.publish(twist_vel)

            self.rl_agent.learnActor()
            self.rl_agent.learnCritic()
            self.rl_agent.clearHistory()

            # path=os.environ["MODEL_PATH"]
            # if path=="":
            #     print("No path defined, model not saved")
            # else:
            #     self.rlDNN.save(path+"model_nav.h5")
            # self.resetWorld()
            self.episode=0
        else:
            self.episode=self.episode+1
        if self.total_eps>self.max_eps:# or grab:
            self.total_eps=0
            self.rl_agent.saveNets()
            self.f.write(str(self.total_episode_reward))
            self.f.write('\n')
            self.f.flush()
            self.total_episode_reward=0
            self.coke_box_state[:]=0
            self.light_box_state[:]=0
            self.gu.resetWorld()
        else:
            self.total_eps=self.total_eps+1

    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()

if __name__=='__main__':
    print("IN METHOD")
    coke_no=int(sys.argv[1])
    light_no=int(sys.argv[2])
    print("STARTING NODE")
    rospy.init_node("robotRL", anonymous=True)
    print("Making robot")
    robot=RobotRL(coke_no, light_no)
    robot.start()