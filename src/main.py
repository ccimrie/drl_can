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
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

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
        if os.path.exists(path+"model6.h5"):
            self.rlDNN=keras.models.load_model(path+"model.h5")
        else:
            num_actions = 5 # linear x mu, angular z mu, linear x var, angular z var, x&z covar
            num_hidden = 128
            # inputs = layers.Input(shape=(480,640,3,))
            inputs = layers.Input(shape=(350,640,3,))
            hl_1 = layers.Conv2D(32, 8, activation="relu")(inputs)
            hl_2 = layers.Conv2D(16, 4, activation="relu")(hl_1)
            hl_flatten=layers.Flatten()(hl_2)
            hl_3 = layers.Dense(64, activation="relu")(hl_flatten)
            action = layers.Dense(num_actions, activation="sigmoid")(hl_3)
            critic = layers.Dense(1)(hl_3)
            self.rlDNN=keras.Model(inputs=inputs, outputs=[action, critic])

        self.optimizer=keras.optimizers.Adam(learning_rate=0.01)
        self.huber_loss = keras.losses.Huber()
        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []
        self.running_reward = 0
        self.episode=0
        # self.tape=tf.GradientTape()
        # episode_count = 0

        # RL Params
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()
        self.vel=[0,0]
        # State variables
        self.image=None
        self.current_vel=[0,0,0]
        self.joints=[0,0,0,0]
        self.total_eps=0
        self.max_eps=250
        self.learn_ep=10

        # Subscribers
        ##   Subscribe vision (image from camera)
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.imageSub)

        ##   Subscribe robot state(current velocity, and arm state)
        # rospy.Subscriber("/cmd_vel", Twist, self.velSub)
        # rospy.Subscriber("/arm_controller/state", , self.velSub)

        # Publishers
        self.vel_pub=rospy.Publisher("/cmd_vel", Twist,queue_size=1)
        self.state_publish=rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)

        ## Initial reset of world to randomise coke can positions
        self.resetWorld()

        print("initialised")

    def gms_client(self,model_name,relative_entity_name):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp1 = gms(model_name,relative_entity_name)
            return resp1
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def imageSub(self, data):
        # Convert image to state for DRL-network
        self.image = self.br.imgmsg_to_cv2(data)[40:390,:]/255.0
        # cv2.imshow('image',self.image)
        # cv2.waitKey(1)
        with tf.GradientTape() as tape:
            state = tf.convert_to_tensor(self.image)#.reshape(480*640*3))#(-1,480,640,3))
            state=tf.expand_dims(state,0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            act_probs, critic_value = self.rlDNN(state)
            action_probs=np.squeeze(act_probs[0])
            # print(action_probs)
            # Send velocity to robot
            x_mu=(2*action_probs[0]-1)
            x_var=action_probs[1]

            az_mu=(2*action_probs[2]-1)
            az_var=action_probs[3]

            rho_var=action_probs[4]
            azx_var=rho_var*np.sqrt(x_var)*np.sqrt(az_var)

            twist_vel=Twist()
            mu=np.array([x_mu,az_mu])
            sigma=np.array([[x_var,azx_var],[azx_var,az_var]])

            act=np.random.multivariate_normal(mu,sigma)
            twist_vel.linear.x=act[0]*0.05
            twist_vel.angular.z=act[1]*0.05
            self.vel_pub.publish(twist_vel)
            time.sleep(0.05)
            
            # Record outcome
            self.critic_value_history.append(critic_value[0, 0])
            p_act_denom=1.0/np.sqrt(np.linalg.det(2*np.pi*sigma)+self.eps)
            p_act_exp=-0.5*np.matmul(np.matmul(np.transpose((act-mu)),np.linalg.pinv(sigma,hermitian=True)),(act-mu))
            p_act=p_act_denom*np.exp(p_act_exp)
            self.action_probs_history.append(np.log(p_act+self.eps))

            # Get reward
            # Reward is distance between coke and robot optimal is 0.2 distance
            ## Assumption: always at least one coke can
            pos_coke=self.gms_client("coke_1","link").pose.position
            pos_robot=self.gms_client("robot", "").pose.position
            dist=(pos_coke.x-pos_robot.x)**2+(pos_coke.y-pos_robot.y)**2
            for i in self.coke_list[1:-1]:
                pos_coke_temp=self.gms_client(i,"link").pose.position
                dist_temp=(pos_coke_temp.x-pos_robot.x)**2+(pos_coke_temp.y-pos_robot.y)**2
                if dist_temp<dist:
                    dist=dist_temp
                    pos_coke=pos_coke_temp
        
            reward=1.0/((np.sqrt(dist)-0.3)**2+1e-2)
        
            self.rewards_history.append(reward)

            # After fixed episode length learn
            if self.episode==self.learn_ep:
                twist_vel.linear.x=0
                twist_vel.angular.z=0
                self.vel_pub.publish(twist_vel)

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
                history = zip(self.action_probs_history, self.critic_value_history, returns)
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
                    actor_losses.append(-log_act * diff)  # actor loss

                    # The critic must be updated so that it predicts a better estimate of
                    # the future rewards.
                    critic_losses.append(self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

                # Backpropagation
                loss_value = sum(actor_losses) + sum(critic_losses)
                grads = tape.gradient(loss_value, self.rlDNN.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.rlDNN.trainable_variables))

                # Reset values
                # Clear the loss and reward history
                self.action_probs_history.clear()
                self.critic_value_history.clear()
                self.rewards_history.clear()
                self.episode=0
            else:
                self.episode=self.episode+1
            if self.total_eps==self.max_eps:
                self.total_eps=0
                path=os.environ["MODEL_PATH"]
                if path=="":
                    print("No path defined, model not saved")
                else:
                    self.rlDNN.save(path+"model.h5")
                self.resetWorld()
            else:
                self.total_eps=self.total_eps+1

    def velSub(self, data):
        self.current_vel=data
        print(self.current_vel)

    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()

    def resetWorld(self):
        # rospy.wait_for_service('/gazebo/reset_world')
        # rospy.wait_for_service('/gazebo/reset_simulation')
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        ## Reset coke position
        # coke_pub=rospy.Publisher("/gazebo_set_model_state", int,queue_size=1)
        # reset_world = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        reset_world()

        ## Dimension is 10x10 ([[-5:5],[-5:5]]) arena (harcoded for now)
        ## Make coke spawn area smaller; easier for camera to see
        for i in self.coke_list:
            self.smsClient(i, (1-2*np.random.random(2))*3)
        for j in self.light_list:
            self.smsClient(i, (1-2*np.random.random(2))*3)

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