'''DDPG Agent'''
import numpy as np
import os
import pandas as pd
from quad_controller_rl import util

from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.Actor import Actor
from quad_controller_rl.agents.Critic import Critic
from quad_controller_rl.agents.OUNoise import OUNoise
from quad_controller_rl.agents.replay_buffer import ReplayBuffer

class DDPG(BaseAgent):
    def __init__(self, task):
        self.task = task

        self.state_size = 3 # position only
        self.action_size = 3 #  force only
        self.action_low = self.task.action_space.low[0:3]
        self.action_high = self.task.action_space.high[0:3]
        print("Original spaces: {}, {}\nConstrained spaces: {}, {}".format(
            self.task.observation_space.shape, self.task.action_space.shape,
            self.state_size, self.action_size))

        #load/save parameters
        self.load_weights = True # try to load weights from previously saved models
        self.save_weights_every = 100 # None to disable
        self.model_dir = util.get_param('out')

        self.model_name = "my-model"
        self.model_ext = ".h5"
        self.episode = 0
        if self.load_weights or self.save_weights_every:
            self.actor_filename = os.path.join(self.model_dir, "{}_actor{}".format(self.model_name, self.model_ext))
            self.critic_filename = os.path.join(self.model_dir, "{}_critic{}".format(self.model_name, self.model_ext))
            print("Actor filename:", self.actor_filename)
            print("Critic filename:", self.critic_filename)

        # Actor(Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic(Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)


        # Load pre-trained model weights, if available
        if self.load_weights and os.path.isfile(self.actor_filename):
            try:
                self.actor_local.model.load_weights(self.actor_filename)
                self.critic_local.model.load_weights(self.critic_filename)
                print("Model weights loaded from file") # [debug]
            except Exception as e:
                print("Unable to load model weights from file!")
                print("{}: {}".format(e.__class__.__name__, str(e)))

        if self.save_weights_every:
            print("Saving model weights", "every {} episodes".format(
                self.save_weights_every) if self.save_weights_every else "disabled")  # [debug]

        

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.noise = OUNoise(self.action_size)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.99 #discount factor
        self.tau = 0.001 # for soft

        self.rewards_list = []
        self.reset_episode_vars()
        
        # Save episode stats
        self.stats_filename = os.path.join(util.get_param('out'), "stats_{}.csv".format(util.get_timestamp()))
        self.stats_columns = ['episode', 'total_reward'] # specify column to save
        self.episode_num = 1
        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename)) # debug
        #print("init complete") #[debug]




    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        self.episode += 1
        
    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions"""
        return state[0:3] # position only

    def postprocess_action(self, action):
        """Return complete action vector"""
        complete_action = np.zeros(self.task.action_space.shape) # shape (6,)
        complete_action[0:3] = action # linear force only
        return complete_action

    def step(self, state, reward, done):
        #print("take a step") #[debug]
        
        # Reduce state vector
        state = self.preprocess_state(state)

        # Choose an action (get action through local actor network)
        action = self.act(state)

        # Save experience/reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)
            self.total_reward += reward
            self.count += 1
        
        # Learn, if replay buffer is ample to sample experiences (online learning)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        if done:
            #print("Done") #[debug]
            self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1

            # Save model weights at regular intervals 
            if self.save_weights_every and self.episode % self.save_weights_every == 0:
                self.actor_local.model.save_weights(self.actor_filename)
                self.critic_local.model.save_weights(self.critic_filename)
                print("Model weights saved at episode", self.episode)  # [debug]
            self.reset_episode_vars()
        
        self.last_state = state
        self.last_action = action
        #print("end of step") #[debug]
        return self.postprocess_action(action)

    # to save rewards stats
    def write_stats(self, stats):
        """Write single episode stats to CSV file"""
        df_stats = pd.DataFrame([stats], columns = self.stats_columns)
        df_stats.to_csv(self.stats_filename, mode='a', index=False, header=not os.path.isfile(self.stats_filename))
        print(stats) # debug

    def learn(self, experiences):
        #print("start learn") #[debug]
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from targets models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model(local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y = Q_targets)

        # Train actor model (local)
        # learning_phase() = 0 -> test mode
        # learning_phase() = 1 -> train mode
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),(-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])

        # Soft-update target models
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)


    def act(self, states):
        #print("act") #[debug]
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        return actions + self.noise.sample() # add some noise for exploration

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.model.get_weights())
        target_weights = np.array(target_model.model.get_weights())

        new_weights = self.tau * local_weights + (1-self.tau) * target_weights
        target_model.model.set_weights(new_weights)
