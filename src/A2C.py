import numpy as np
import math
import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import compat
import tensorflow_probability as tfp
import random
from collections import deque
import os

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class A2C:
    def __init__(self, _num_actions, _state_size):
        ## RL params
        self.gamma=0.99
        # self.optimizer=keras.optimizers.Adam(lr=5e-6)
        # lr=5e-6
        # self.optimizer=keras.optimizers.legacy.Adam(learning_rate=5e-6)
        self.optimizer=keras.optimizers.legacy.Adam(learning_rate=0.05)
        self.huber_loss = keras.losses.Huber()
        self.num_actions=_num_actions
        self.state_size=_state_size

        self.path=os.environ["MODEL_PATH"]
        if os.path.exists(self.path+"actor_model.h5"):
            print("Loading models")
            self.actor=keras.models.load_model(self.path+"actor_model.h5")
            self.critic=keras.models.load_model(self.path+"critic_model.h5")
        else:
            print("Creating models")
            ## Create actor model
            self.actor=self.createActorModel()

            ## Create critic model
            self.critic=self.createCriticModel()

        ## Memory
        self.state_history=[]
        self.action_history=[]
        self.value_pred_history=[]
        self.reward_history=[]

######## Creating DNN models for DRL #######
    def createActorModel(self):
        ## Create actor network
        inputs = layers.Input(shape=(self.state_size,))
        
        hl_1 = layers.Dense(256, activation="relu")(inputs)
        hl_2 = layers.Dense(256, activation="relu")(hl_1)
        hl_3 = layers.Dense(256, activation="relu")(hl_2)
        hl_4 = layers.Dense(256, activation="relu")(hl_3)
        hl_5 = layers.Dense(256, activation="relu")(hl_4)
        hl_6 = layers.Dense(256, activation="relu")(hl_5)

        mu = layers.Dense(self.num_actions, activation="tanh")(hl_6)
        sigma = layers.Dense(self.num_actions, activation="softplus")(hl_6)
        actor_net=keras.Model(inputs=inputs, outputs=[mu, sigma])
        ##actor_net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=[tf.keras.metrics.CategoricalAccuracy()])
        return actor_net

    def createCriticModel(self):
        ## Create critic model
        inputs = layers.Input(shape=(self.state_size,))

        hl_1 = layers.Dense(256, activation="relu")(inputs)
        hl_2 = layers.Dense(256, activation="relu")(hl_1)
        hl_3 = layers.Dense(256, activation="relu")(hl_2)
        hl_4 = layers.Dense(256, activation="relu")(hl_3)
        ##hl_5 = layers.Dense(256, activation="relu")(hl_4)
        ##hl_6 = layers.Dense(256, activation="relu")(hl_5)

        critic = layers.Dense(1)(hl_4)
        critic_net=keras.Model(inputs=inputs, outputs=critic)
        return critic_net

####### Step in the world ######
    ## Assume sigma is diagonal covariance
    def multiVariateDistSample(self, mu, sigma):
        tfd=tfp.distributions
        mvn=tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        act=mvn.sample()
        #print("Sample MU & Sigma")
        #print(mu, sigma)
        return act

    def multiVariateDistLog(self, mu, sigma, act):
        tfd=tfp.distributions
        mvn=tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        #print("Prob MU & Sigma")
        #print(mu, sigma)
        log_prob=mvn.log_prob(act)
        return log_prob

    ## State needs to be tensor
    def tdTarget(self, reward, next_state):
        value=self.critic(next_state)[0,0]
        y_target=self.reward+self.gamma*value
        return y_target

    def step(self, _state):
        ##self.memory=_state
        self.state_history.append(_state)
        state = tf.convert_to_tensor(_state)
        state=tf.expand_dims(state,0)
        
        ## Get Gaussian parameters from actor network
        act_probs, sigma_out=self.actor(state)
        action_probs=np.squeeze(act_probs[0])
        sigma_out=np.squeeze(sigma_out[0])
        sigma_modded=sigma_out+1e-5

        ## Get action sample from Guassian
        act=self.multiVariateDistSample(action_probs, sigma_modded)
        ## Get value from critic network
        value=self.critic(state)[0,0]

        ## Record values
        self.action_history.append(act)
        self.value_pred_history.append(value)

        return np.array(act)

    def recordReward(self, reward):
        self.reward_history.append(reward)

###### Learning #######
    def actorLossFunc(self,log_prob, adv):
        loss=tf.convert_to_tensor(-log_prob*adv)
        return loss

    def calcValue(self):
        rets=[]
        discounted_sum=0
        for r in self.reward_history[::-1]:
            discounted_sum=r+self.gamma*discounted_sum
            rets.insert(0, discounted_sum)
        return rets

    ## Need to be done before critic learning due to TD
    def learnActor(self):
        print("Training actor network")
        returns_true=self.calcValue()
        history=zip(self.state_history, self.action_history, self.value_pred_history, returns_true)
        with tf.GradientTape() as tape:
            actor_loss=0
            for state, act, value, ret in history:
                state=tf.convert_to_tensor(state)
                state=tf.expand_dims(state,0)
                act_probs, sigma_out=self.actor(state)
                
                action_probs=act_probs[0]
                sigma_out=sigma_out[0]
                sigma_modded=sigma_out+1e-5

                ## Get action sample from Guassian
                log_prob=self.multiVariateDistLog(action_probs, sigma_modded, act)

                ## Get action log prob
                adv=ret-value
                actor_loss=actor_loss+(-log_prob*adv)
        #print(actor_loss, act, prob, log_prob, adv)
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        # print("Trained actor network")

    ## Needs next state
    def learnCritic(self):
        print("Training critic network")
        returns_true=self.calcValue()
        history=zip(self.state_history, returns_true)
        with tf.GradientTape() as tape:
            critic_loss=0
            for state, ret in history:
                state=tf.convert_to_tensor(state)
                state=tf.expand_dims(state,0)
                value=self.critic(state)[0,0]
                critic_loss=critic_loss+self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
        #print(critic_loss)
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        # print("Trained critic network")

    def clearHistory(self):
        self.state_history=[]
        self.action_history=[]
        self.value_pred_history=[]
        self.reward_history=[]

##### Exploit ######
    def act(self, _state):
        state = tf.convert_to_tensor(_state)
        state=tf.expand_dims(state,0)
        
        ## Get mu from actor network
        act_vals, _=self.actor(state)
        action_vals=np.squeeze(act_vals[0])
        print(action_vals)
        return action_vals

    def saveNets(self):
        path=os.environ["MODEL_PATH"]
        print("Attempting to save networks")
        if path=="":
            print("No path defined, model not saved")
        else:
            self.actor.save(path+"actor_model.h5")
            self.critic.save(path+"critic_model.h5")
            print("Succesfully saved networks!")