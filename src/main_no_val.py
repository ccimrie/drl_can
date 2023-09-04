#!/usr/bin/env python3
import numpy as np
import rospy
# import roslib; roslib.load_manifest('gazebo')
import time
import cv2
from cv_bridge import CvBridge

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
from turtlebot_movement import TurtlebotMovement as NAV
from robot_arm import RobotArm

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
        self.state_size=1
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

        laser_state_size=360
        input_state_size=2*self.width_state_size*self.height_state_size+laser_state_size
        self.rl_agent=A2C(2, input_state_size)
        print("Success!")
        self.episode=0
        self.learn_eps=1000
        self.total_eps=0

        eps_no=1
        self.max_eps=self.learn_eps*eps_no

        if os.path.isfile('rewards.txt'):
            self.f=open('rewards.txt', 'a')
        else:
            self.f=open('rewards.txt', 'w+')

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
        ## reset arm
        self.arm.moveArmHome()

        ## Publishers
        self.vel_pub=rospy.Publisher("/cmd_vel", Twist,queue_size=1)
        self.arm_update_pub=rospy.Publisher("/arm_motion",Int8,queue_size=1)
        print("Publishers setup")

        self.simReset()

        ## Subscribers
        # rospy.Subscriber("/camera/rgb/image_raw", Image, self.imageSub)
        rospy.Subscriber("/arm_motion", Int8, self.armUpdate, queue_size=1)
        ## Need to get number of objects detected, if >0 see what latest bounding boxes are
        self.state_perception_sub=rospy.Subscriber("/darknet_ros/found_object", ObjectCount, self.imageSub, queue_size=1)
        self.bounding_boxes_sub=rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.storeBoxes, queue_size=1)

        self.laser_sub=rospy.Subscriber('/scan', LaserScan, self.align, queue_size=1)
        print("Subscribers setup")

    def armUpdate(self, data):
        print("Changing arms!")
        arm_temp=data.data
        if arm_temp==0:
            print("Arm change: check for reward")
            twist=Twist()
            twist.linear.x=0.0
            twist.angular.z=0.0
            self.vel_pub.publish(twist)
            pos_coke=self.gu.gms_client("coke_1","link").pose.position
            ## Assume last reward step was for grab attempt
            if pos_coke.z>0.1:
                reward=10
            else:
                reward=-10
            self.rl_agent.recordReward(reward)
            self.total_episode_reward+=reward
            self.trainAgent()
            self.episodeEnd()
            print("Resubscribing...")
            self.state_perception_sub=rospy.Subscriber("/darknet_ros/found_object", ObjectCount, self.imageSub, queue_size=1)
            self.bounding_boxes_sub=rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.storeBoxes, queue_size=1)
            print("Success!")

    def trainAgent(self):
        self.rl_agent.learnActor()
        self.rl_agent.learnCritic()
        self.rl_agent.clearHistory()

    def storeBoxes(self,data):

        ##self.coke_box_state
        width=640.0
        height=480.0

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

    ## Only consider cans in view
    def checkCoke(self):
        raspi_cam_angle=(np.pi/180.0)*(62.2/2.0) # radians
        robot_pos=self.gu.gms_client('robot','link')
        rotation_angle=2*np.arcsin(robot_pos.pose.orientation.z)
        dist=-1
        for coke in self.coke_list:
            # print(coke)
            coke_pos=self.gu.gms_client(coke,'link')
            # print(coke_pos)
            x=coke_pos.pose.position.x-robot_pos.pose.position.x
            y=coke_pos.pose.position.y-robot_pos.pose.position.y
            x_new=x*np.cos(-rotation_angle)-y*np.sin(-rotation_angle)
            y_new=x*np.sin(-rotation_angle)+y*np.cos(-rotation_angle)
            width_bound=x_new*np.tan(raspi_cam_angle)
            if abs(y_new)<width_bound:
                dist_temp=np.sqrt(x_new**2+y_new**2)
                if dist==-1 or dist_temp<dist:
                    dist=dist_temp
        return dist

    def calculateReward(self):
        ## Check if system can 'see' cans
        state=self.coke_box_state.copy()
        reward_observe=(np.sum(state>0))
        if reward_observe>0:
            reward_cell=np.sum(state>0.75)/self.state_size
            reward_dist=self.checkCoke()
            ideal_dist=0.4
            if not(reward_dist==-1):
                reward_dist=(1.0/((reward_dist-ideal_dist)**2+1.0))
            else:
                reward_dist=0
            # print(reward_dist, reward_cell)
            return reward_dist#+reward_cell
        else:
            return 0

    def laserScanStore(self):
        self.laser=data.ranges
    
    def imageSub(self, data):
        return
        # print("IMAGES")
        if self.episode%100==0:
            print(self.episode)

        # Get state
        image_coke=self.coke_box_state.flatten()
        image_light=self.light_box_state.flatten()
        laser_state=self.laser/3.5 ## max range of scan in metres
        image=np.append(image_coke,image_light)
        state=np.append(image, laser_state)

        # image_scaled=cv2.resize(self.coke_box_state, dsize=(640,480), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('test',image_scaled)
        # cv2.waitKey(1)

        ## Send state to the networks and get actions
        act=self.rl_agent.step(image)

        act[0]=np.clip(act[0],-1,1)
        act[1]=np.clip(act[1],-1,1)

        ## Check if a grab or move action:
        # p=np.random.random()
        twist_vel=Twist()
        
        grab_thresh=0.2
        if abs(act[0])>grab_thresh or abs(act[1])>grab_thresh:
        # if True:
            twist_vel.linear.x=act[0]*0.2
            twist_vel.angular.z=act[1]*0.15
            self.vel_pub.publish(twist_vel)
        else:
            print(act)
            twist_vel.linear.x=0.0
            twist_vel.angular.z=0.0
            self.vel_pub.publish(twist_vel)
            reward=self.calculateReward()
            # self.rl_agent.recordReward(reward)
            self.episode=self.episode+1
            print("Unsubscribing")
            self.state_perception_sub.unregister()
            self.bounding_boxes_sub.unregister()
            # cv2.destroyAllWindows()
            self.arm_update_pub.publish(1)
            return

        ## Add 'state decay'
        memory_thresh=0.2
        self.coke_box_state=self.coke_box_state*0.9
        self.light_box_state=self.light_box_state*0.9
        self.coke_box_state[self.coke_box_state<memory_thresh]=0
        self.light_box_state[self.light_box_state<memory_thresh]=0

        rospy.sleep(0.1)

        twist_vel.linear.x=0.0
        twist_vel.angular.z=0.0
        self.vel_pub.publish(twist_vel)

        ## Get reward
        ## Check for collisions
        if self.gu.collision==True:
            reward=-5
            self.rl_agent.recordReward(reward)
            self.total_episode_reward+=reward
            self.trainAgent()
            self.episodeEnd()
            return
        else:
            reward=self.calculateReward()        

        #### TODO
        ## Check for collision
        terminate=0

        self.rl_agent.recordReward(reward)
        self.total_episode_reward+=reward

        ## After fixed episode length learn
        ## (should also learn if robot completes dropping off?)
        if self.episode>self.learn_eps:
            twist=Twist()
            twist_vel.linear.x=0.0
            twist_vel.angular.z=0.0
            self.vel_pub.publish(twist_vel)

            self.trainAgent()
            self.rl_agent.saveNets()
            self.episode=0
        else:
            self.episode=self.episode+1
        if self.total_eps>self.max_eps or terminate:# or grab:
            self.episodeEnd()
        else:
            self.total_eps=self.total_eps+1

    def episodeEnd(self):
        self.total_eps=0
        self.episode=0
        self.rl_agent.saveNets()
        self.f.write(str(self.total_episode_reward))
        self.f.write('\n')
        self.f.flush()
        self.total_episode_reward=0
        self.coke_box_state[:]=0
        self.light_box_state[:]=0
        self.simReset()

    def simReset(self):
        # self.gu.resetWorldTest()
        self.gu.resetWorld(self.coke_list, self.light_list)
        self.arm.moveArmHome()
        rospy.sleep(0.5)
        print("World reset!")

    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()

if __name__=='__main__':
    coke_no=int(sys.argv[1])
    light_no=int(sys.argv[2])
    rospy.init_node("robotRL", anonymous=True)
    robot=RobotRL(coke_no, light_no)
    robot.start()