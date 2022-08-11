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

class RobotRL(object):

    def __init__(self, coke_no):
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

        # Network
        # num_inputs = 480*640*3
        ## Check if a model is already created
        path=os.environ["MODEL_PATH"]
        if os.path.exists(path+"model_nav.h5"):
            print("Loading model")
            self.rlDNN=keras.models.load_model(path+"model_nav.h5")
        else:
            print("Creating new neural network")
            num_actions = 2 # linear x mu, angular z mu, linear x var, angular z var; no x&z covar or grab
            num_hidden = 128
            # inputs = layers.Input(shape=(480,640,3,))
            inputs = layers.Input(shape=(350,640,3,))
            hl_1 = layers.Conv2D(8, 7, activation="relu")(inputs)
            hl_2 = layers.Conv2D(16, 5, activation="relu")(hl_1)
            # hl_3 = layers.Conv2D(32, 3, activation="relu")(hl_2)
            hl_flatten=layers.Flatten()(hl_2)
            hl_4 = layers.Dense(64, activation="relu")(hl_flatten)
            hl_5 = layers.Dense(64, activation="relu")(hl_4)
            mu = layers.Dense(num_actions, activation="sigmoid")(hl_5)
            sigma = layers.Dense(num_actions, activation="softplus")(hl_5)
            critic = layers.Dense(1)(hl_5)
            self.rlDNN=keras.Model(inputs=inputs, outputs=[mu, sigma, critic])

        self.optimizer=keras.optimizers.Adam(learning_rate=0.01)
        self.huber_loss = keras.losses.Huber()
        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []
        self.running_reward = 0
        self.image_history=[]

        self.sanity=[]

        # RL Params
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()
        self.episode=0
        self.total_eps=0
        self.learn_eps=1000
        self.max_eps=self.learn_eps*2

        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("arm")
        self.gripper_group=moveit_commander.MoveGroupCommander("gripper")    
        self.grab_attempts=0
        self.arm=0

        ## Subscribers
        ## Subscribe vision (image from camera)
        ## Subscribe to arm motion, determines whether navigating or grabbing can
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.imageSub)
        rospy.Subscriber("/arm_motion", Int8, self.armUpdate)

        ## Publishers
        self.vel_pub=rospy.Publisher("/cmd_vel", Twist,queue_size=1)
        self.arm_update_pub=rospy.Publisher("/arm_motion",Int8,queue_size=1)
        
        ## Initial reset of world to randomise coke can positions
        self.resetWorld()
        self.gripperOpen()
        # self.moveArmAngle([1.2,-0.4,-0.8])

        # self.moveArm([0.182,0.0,0.281])
        # print(self.move_group.get_current_pose())
        # self.moveArmPose([0.2,0.0,0.17])
        # self.gripperClose()
        # self.moveArmPose([0.182,0.0,0.281])
        # print("initialised")

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
                path=os.environ["MODEL_PATH"]
                if path=="":
                    print("No path defined, model not saved")
                else:
                    self.rlDNN.save(path+"model_nav.h5")
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

    def imageSub(self, data):
        if self.arm==1:
            return

        if self.episode%100==0:
            print(self.episode)

        # Convert image to state for DRL-network
        image = self.br.imgmsg_to_cv2(data)[90:440,:]/255.0
        self.image_history.append(image)
        
        state = tf.convert_to_tensor(image)#.reshape(480*640*3))#(-1,480,640,3))
        state=tf.expand_dims(state,0)

        # Predict action probabilities and estimated future rewards
        # from environment state
        act_probs, sigma_out, critic_value = self.rlDNN(state)
        action_probs=np.squeeze(act_probs[0])
        sigma_out=np.squeeze(sigma_out[0])
        
        self.sanity.append([act_probs, sigma_out, critic_value])

        ## Check if a grab or move action:
        # p=np.random.random()
        twist_vel=Twist()
        
        # Send velocity to robot
        x_mu=(2*action_probs[0]-1)
        x_var=sigma_out[0]

        az_mu=(2*action_probs[1]-1)
        az_var=sigma_out[1]

        ## Ignore off-diagonals in covar matrix?
        azx_var=0.0

        twist_vel=Twist()
        mu=np.array([x_mu,az_mu])
        sigma=np.array([[x_var,azx_var],[azx_var,az_var]])

        act=np.random.multivariate_normal(mu,sigma)

        act=np.clip(act,-1,1)

        # print("\n\nact/mu/sigma: ", act, mu, sigma)

        # twist_vel.linear.x=act[0]*0.05
        # twist_vel.angular.z=act[1]*0.05
        # self.vel_pub.publish(twist_vel)
        # time.sleep(0.2)

        ## Grab if speed is less than 0.01?
        # print('\n\n',act[0],act[1])
        if np.abs(act[0])>0.1 or np.abs(act[1])>0.1:
            twist_vel.linear.x=act[0]*0.1
            twist_vel.angular.z=act[1]*0.1
            self.vel_pub.publish(twist_vel)
            # time.sleep(0.)
        else:
            twist_vel.linear.x=0.0
            twist_vel.angular.z=0.0
            self.vel_pub.publish(twist_vel)

            eig_values=np.linalg.eig(2*np.pi*sigma)[0]
            pd=np.product(eig_values[eig_values>1e-12])
            p_act_denom=1.0/np.sqrt(pd)
            p_act_exp=-0.5*np.matmul(np.matmul(np.transpose((act-mu)),np.linalg.pinv(sigma,hermitian=True)),(act-mu))
            p_act=p_act_denom*np.exp(p_act_exp)
            self.action_probs_history.append(p_act)            

            if p_act<0:
                print("oops")
            elif p_act>1:
                print("Not right")
                print("Act/sigma/Mu: ", act, sigma, mu)

            pos_coke=self.gms_client("coke_1","link").pose.position
            pos_robot=self.gms_client("robot", "").pose.position
            dist=(pos_coke.x-pos_robot.x)**2+(pos_coke.y-pos_robot.y)**2
            reward=1.0/((np.sqrt(dist)-0.35)**2+1e-1)
            self.rewards_history.append(reward)
            self.episode=self.episode+1
            self.arm_update_pub.publish(1)
            return

        # Record outcome
        # self.critic_value_history.append(critic_value[0, 0])
        # p_act_denom=1.0/np.sqrt(np.linalg.det(2*np.pi*sigma)+self.eps)

        ## Calculate the pseudo-determinent
        eig_values=np.linalg.eig(2*np.pi*sigma)[0]
        pd=np.product(eig_values[eig_values>1e-12])
        p_act_denom=1.0/np.sqrt(pd)
        p_act_exp=-0.5*np.matmul(np.matmul(np.transpose((act-mu)),np.linalg.pinv(sigma,hermitian=True)),(act-mu))
        p_act=p_act_denom*np.exp(p_act_exp)
        if p_act<0:
            print("oops")
        elif p_act>1:
            print("Not right")
            print("Act/sigma/Mu: ", act, sigma, mu)
        self.action_probs_history.append(p_act)

        # Get reward
        # Reward is distance between coke and robot optimal is 0.2 distance
        ## Assumption: always at least one coke can
        pos_coke=self.gms_client("coke_1","link").pose.position
        pos_robot=self.gms_client("robot", "").pose.position
        dist=(pos_coke.x-pos_robot.x)**2+(pos_coke.y-pos_robot.y)**2
        # for i in self.coke_list[1:-1]:
            #     pos_coke_temp=self.gms_client(i,"link").pose.position
        #     dist_temp=(pos_coke_temp.x-pos_robot.x)**2+(pos_coke_temp.y-pos_robot.y)**2
        #     if dist_temp<dist:
        #         dist=dist_temp
        #         pos_coke=pos_coke_temp
    
        reward=1.0/((np.sqrt(dist)-0.35)**2+1e-1)

        self.rewards_history.append(reward)
        # After fixed episode length learn
        if self.episode>self.learn_eps:
            twist=Twist()
            twist_vel.linear.x=0.0
            twist_vel.angular.z=0.0
            self.vel_pub.publish(twist)
            self.imageSubLearn()
            path=os.environ["MODEL_PATH"]
            if path=="":
                print("No path defined, model not saved")
            else:
                self.rlDNN.save(path+"model_nav.h5")
            self.resetWorld()
            self.episode=0
        else:
            self.episode=self.episode+1
        # if self.total_eps>self.max_eps or grab:
        #     self.total_eps=0
        # #     # path=os.environ["MODEL_PATH"]
        # #     # if path=="":
        # #         # print("No path defined, model not saved")
        # #     # else:
        # #         # self.rlDNN.save(path+"model_grab.h5")
        #     self.resetWorld()
        # else:
        #     self.total_eps=self.total_eps+1

    def imageSubLearn(self):
        print("\n\nLearning\n\n")
        act_probs_history=[]
        critic_value_history=[]

        print("image list: ", np.shape(self.image_history))
        print("act list: ", np.shape(self.action_probs_history))

        with tf.GradientTape() as tape:
            ## Re-observe input
            for i in np.arange(len(self.image_history)):
                state = tf.convert_to_tensor(self.image_history[i])#.reshape(480*640*3))#(-1,480,640,3))
                state=tf.expand_dims(state,0)

                # Predict action probabilities and estimated future rewards
                # from environment state
                act_probs, sigma_out, critic_value = self.rlDNN(state)

                action_probs=np.squeeze(act_probs[0])
                sigma_out=np.squeeze(sigma_out[0])
            
                ## Check if a grab or move action:
                # Send velocity to robot
                x_mu=(2*action_probs[0]-1)
                x_var=sigma_out[0]

                az_mu=(2*action_probs[1]-1)
                az_var=sigma_out[1]

                ## Ignore off-diagonals in covar matrix?
                azx_var=0.0

                mu=np.array([x_mu,az_mu])
                sigma=np.array([[x_var,azx_var],[azx_var,az_var]])

                act=self.action_probs_history[i] ##np.random.multivariate_normal(mu,sigma)
                
                # Record outcome
                critic_value_history.append(critic_value[0, 0])
                if act==1:
                    act_probs_history.append(np.log(act))
                else:
                    act_probs_history.append(np.log(act+self.eps))
        
            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in self.rewards_history[::-1]:
                discounted_sum = r + self.gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
            returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(act_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            # print(self.action_probs_history)
            for log_act, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                print(diff, ret, value, log_act, '\n\nActor loss: ', -log_act*diff)
                actor_losses.append(-log_act * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, self.rlDNN.trainable_variables)
            print("Acquired gradients")
            self.optimizer.apply_gradients(zip(grads, self.rlDNN.trainable_variables))
            print("Learning completed\n\n")
            # Reset values
            self.image_history.clear()
            self.rewards_history.clear()
            self.action_probs_history.clear()
            self.sanity.clear()
            self.episode=0

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
        
    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()

    def resetWorld(self):
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        ## Reset coke position
        reset_world()
        self.gripperOpen()
        self.moveArmHome()
        ## Dimension is 10x10 ([[-5:5],[-5:5]]) arena (harcoded for now)
        ## Make coke spawn area smaller; easier for camera to see
        # for i in self.coke_list:
            # self.smsClient(i, (1-2*np.random.random(2))*3)

    def getCoke(self, name):
        model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        # print(model_coordinates(name, "link"))
        return model_coordinates(name, "link")

    def smsClient(self, model_name, pos):
        state_msg = ModelState()
        state_msg.model_name = model_name
        state_msg.pose.position.x = pos[0]
        state_msg.pose.position.y = pos[1]
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )

        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

if __name__=='__main__':
    coke_no=int(sys.argv[1])
    rospy.init_node("robotRL", anonymous=True)
    robot=RobotRL(coke_no)
    robot.start()