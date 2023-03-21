#!/usr/bin/env python2
import numpy as np
import rospy
# import roslib; roslib.load_manifest('gazebo')
import time
from cv_bridge import CvBridge
import cv2

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
import moveit_commander
import moveit_msgs.msg

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *

from math import pi

import sys
import os

class RobotRL(object):

    def __init__(self, coke_no, light_no):
        self.loop_rate = rospy.Rate(1)
        self.br = CvBridge()

        # Ros service for getting model states
        self.model_coordinates=rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

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
        self.state_size=10
        width=640
        height=480

        self.width_state_size=int(width/self.state_size)
        self.height_state_size=int(height/self.state_size)

        self.box_state=np.zeros([self.height_state_size, self.width_state_size])

        # Network
        # num_inputs = 480*640*3
        ## Check if a model is already created
        path=os.environ["MODEL_PATH"]

        self.rl_agent=A2C(self.width_state_size*self.height_state_size, 2)

        ## For noetic/python3
        ##self.optimizer=keras.optimizers.Adam(learning_rate=0.01)
        ## For melodic/python2
        self.optimizer=keras.optimizers.Adam(lr=0.01)

        # arm=RobotArm()

        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("arm")
        self.gripper_group=moveit_commander.MoveGroupCommander("gripper")    
        self.grab_attempts=0
        self.arm=0

        ## Subscribers
        ## Subscribe vision (image from camera)
        ## Subscribe to arm motion, determines whether navigating or grabbing can
        ## rospy.Subscriber("/camera/rgb/image_raw", Image, self.imageSub)
        rospy.Subscriber("/arm_motion", Int8, self.armUpdate)

        ## Need to get number of objects detected, if >0 see what latest bounding boxes are
        rospy.Subscriber("/darknet_ros/found_object", ObjectCount, self.imageSub)
        rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.storeBoxes)

        ## Publishers
        self.vel_pub=rospy.Publisher("/cmd_vel", Twist,queue_size=1)
        # self.arm_update_pub=rospy.Publisher("/arm_motion",Int8,queue_size=1)
        
        ## Set move base initial position
        # Tell the action client that we want to spin a thread by default
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.init_pose=rospy.Publisher("/initialpose",geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=1)
        rospy.loginfo("Wait for the action server to come up")
        self.resetWorld()
        self.gripperOpen()
        pos_robot=self.gms_client("robot", "").pose.position
        ont_robot=self.gms_client("robot","").pose.orientation
        self.setInitialPose([pos_robot.x,pos_robot.y,pos_robot.z],[ont_robot.z,ont_robot.w])
        print("Position set")
        # Allow up to 5 seconds for the action server to come up
        self.move_base.wait_for_server(rospy.Duration(5))
        print("Got movebase server")

    def armUpdate(self, data):
        arm_temp=data.data
        print(self.arm)
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

                self.resetWorld()
            else:
                self.rewards_history[-1]=self.rewards_history[-1]-5
                self.moveArmHome()
        self.arm=arm_temp

    def gms_client(self,model_name,relative_entity_name):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp1 = gms(model_name,relative_entity_name)
            return resp1
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def storeBoxes(self,data):
        self.box_state[:]=0
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
            else:
                val=2
            for x in np.arange(xmin, xmax+1):
                for y in np.arange(ymin, ymax+1):
                    if self.box_state[y,x]==0:
                        self.box_state[y,x]=val
                    elif self.box_state[y,x]!=val:
                        self.box_state[y,x]=3

    def imageSub(self, data):
        if self.arm==1:
            return

        if self.episode%100==0:
            print(self.episode)

        # Get state         
        image=self.box_state.flatten()

        ## Send state to the networks and get actions
        act=rl_agent.step(state)

        ## Check if a grab or move action:
        # p=np.random.random()
        twist_vel=Twist()
        
        if True:
            twist_vel.linear.x=act[0]*0.1
            twist_vel.angular.z=act[1]*0.1
            self.vel_pub.publish(twist_vel)
        else:
            twist_vel.linear.x=0.0
            twist_vel.angular.z=0.0
            self.vel_pub.publish(twist_vel)

            pos_robot=self.gms_client("robot", "").pose.position
            reward=0

            for i in np.arange(self.coke_no):
                pos_coke=self.gms_client("coke_"+str(i),"link").pose.position
                dist=(pos_coke.x-pos_robot.x)**2+(pos_coke.y-pos_robot.y)**2
                reward_temp=1.0/((np.sqrt(dist)-0.35)**2+1e-1)

            self.rewards_history.append(reward)
            self.episode=self.episode+1
            self.arm_update_pub.publish(1)
            return

        time.sleep(0.1)

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
        
        reward_observe=np.sum(image==1)
        reward_dist=1.0/((np.sqrt(dist)-0.35)**2+1e-1)

        reward=reward_dist+reward_observe
        self.rewards_history.append(reward)

        ## After fixed episode length learn
        ## (should also learn if robot completes dropping off?)
        if self.episode>self.learn_eps:
            twist=Twist()
            twist_vel.linear.x=0.0
            twist_vel.angular.z=0.0
            self.vel_pub.publish(twist)

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
        # #     # path=os.environ["MODEL_PATH"]
        # #     # if path=="":
        # #         # print("No path defined, model not saved")
        # #     # else:
        # #         # self.rlDNN.save(path+"model_grab.h5")
            self.resetWorld()
        else:
            self.total_eps=self.total_eps+1

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

    def moveArmDown(self):
        pose=self.move_group.get_current_pose().pose
        # pose_goal = geometry_msgs.msg.Pose()
        pose.position.z=pose.position.z-0.1
        move_group.set_pose_target(pose)
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = self.move_group.go(wait=True)
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

    def resetWorld(self):
        print("Resetting world!")
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        ## Reset coke/light position
        reset_world()
        self.gripperOpen()
        self.moveArmHome()

        ## Set robot's position
        self.smsClient('robot', [3.5, 3.5],-1.57)
        pos_robot=self.gms_client("robot", "").pose.position
        ont_robot=self.gms_client("robot","").pose.orientation
        self.setInitialPose([pos_robot.x,pos_robot.y,pos_robot.z],[ont_robot.z,ont_robot.w])
        ## Dimension is 10x10 ([[-5:5],[-5:5]]) arena (harcoded for now)
        ## Make coke spawn area smaller; easier for camera to see
        for i in self.coke_list:
            self.smsClient(i, (1-2*np.random.random(2))*3,(1-2*np.random.random())*np.pi)
        for j in self.light_list:
            self.smsClient(j, (1-2*np.random.random(2))*3,0)

    def smsClient(self, model_name, pos, orient):
        state_msg = ModelState()
        state_msg.model_name = model_name
        state_msg.pose.position.x = pos[0]
        state_msg.pose.position.y = pos[1]
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = orient
        state_msg.pose.orientation.w = 1.0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )

        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

if __name__=='__main__':
    coke_no=int(sys.argv[1])
    light_no=int(sys.argv[2])
    rospy.init_node("robotRL", anonymous=True)
    robot=RobotRL(coke_no, light_no)
    robot.start()