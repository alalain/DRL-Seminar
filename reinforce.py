from model import ReinforcePolicy
from replayBuffer import ReplayBuffer

import tensorflow_probability as tfp

import numpy as np
import time
import gymnasium as gym
import tensorflow as tf
from typing import Tuple
import cv2
import wandb

checkpoint_name = 'checkpoint_ant_4'


class ReinforceAgent:

    def __init__(self,
                 env: gym.Env,
                 val_env: gym.Env,
                 num_epochs: int = 3000,
                 gamma_survive: float = 0.99995,
                 gamma: float = 0.99,
                 learning_rate: float = 3e-4,
                 exploration_rate: float = 0.99,
                 exploration_rate_update: float = 0.999,
                 min_exploration: float = 0.1,
                 log_std_min=-3.5,
                 custom_reward=False,
                 log_std_max=2.5):

        """

        Parameters
        ----------
        log_std_min : 
           min log std for the policy 
        custom_reward : 
           Enable custom reward, if false, reward from env is used 
        log_std_max : 
           max log std for the policy 
        env : gym.Env
           Environment to train on 
        val_env : gym.Env
           environment to validate on 
        num_epochs : int
           number of training epochs 
        gamma_survive : float
           exponential decay weight for survive reward, only valid if custom_reward=True 
        gamma : float
           reward gamma 
        learning_rate : float
           learning rate for the actor 
        exploration_rate : float
           initial exploration rate 
        exploration_rate_update : float
           exponential decay param for the exploration rate 
        min_exploration : float
            minimal expoloration rate

        """
        self.env = env
        self.val_env = val_env

        self.num_epochs = num_epochs

        self.n_steps = tf.Variable(0)
        self.min_exploration = tf.Variable(min_exploration)
        self.is_test = False
        self.gamma = tf.Variable(gamma)

        self.gamma_survive = gamma_survive
        self.current_gamma_survive = gamma_survive
        self.custom_reward = custom_reward

        self.num_observations = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        self.actor = ReinforcePolicy(self.num_actions,
                                     self.num_observations,
                                     log_std_min,
                                     log_std_max,
                                     action_scale=1)  #ANTZy
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate)
        # self.actor_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        self.exploration_rate = tf.Variable(exploration_rate)
        self.exploration_rate_update = exploration_rate_update

    @tf.function()
    def run_policy(self, state):
        """
        Tf function to speed up policy

        Parameters
        ----------
        state : 
           batch containg stets to run policy on 

        Returns
        -------
            
        """
        selected_action, log_prob, mu, std, dist, det_action, z = self.actor(
            state, training=False)

        return selected_action, log_prob, dist, det_action

    @tf.function()
    def select_action(self, state, exploration_rate):

        """

        Parameters
        ----------
        state : 
            current state(s)
        exploration_rate : 
            exploration rate at which random actions are sampled

        Returns
        -------
        
            
        """
        selected_action, log_prob, mu, std, dist, det, z = self.actor(
            state, training=False)
        if tf.random.uniform(shape=[], minval=0.,
                             maxval=1.) < exploration_rate:
            action = tf.random.uniform(shape=[8], minval=-1., maxval=1.)
            action = action * 3
            log_prob = tf.math.reduce_mean(dist.log_prob(selected_action))

        return selected_action, log_prob, z

    def take_step(self, state) -> Tuple[np.ndarray, float, bool]:

        """
        Take a step in the environment

        Parameters
        ----------
        state : 
            current state

        Returns
        -------
        Tuple[np.ndarray, float, bool]
            

        """
        if self.is_test is False:
            selected_action, log_prob, z = self.select_action(
                tf.convert_to_tensor(state),
                tf.maximum(self.exploration_rate, self.min_exploration))
            selected_action = np.squeeze(selected_action.numpy(), axis=0)
            z = np.squeeze(z.numpy(), axis=0)
            next_state, reward, done, truncated, info = self.env.step(
                selected_action)
            if self.custom_reward:
                custom_reward = info[
                    'reward_survive'] * self.current_gamma_survive + info[
                        'reward_forward'] + info['reward_ctrl']
                self.current_gamma_survive *= self.gamma_survive
                reward = custom_reward
            # reward = reward

            return next_state, reward, done, np.squeeze(
                log_prob.numpy()), selected_action, truncated, z

        action, *_, action_det = self.run_policy(tf.convert_to_tensor(state))
        action = np.squeeze(action.numpy(), axis=0)
        next_state, reward, done, truncated, info = self.val_env.step(action)
        return next_state, reward, done

    # @tf.function(reduce_retracing=True)
    @tf.function()
    def update_model(self, states, actions, reward, num_valid):
        """
        Update model based on the calculated losses
        batches will allways be 1000 long even if less steps were made.
        The truncation is done with the num_valid parameter. THis is due to
        the TF tracing which doesn't like input variance in the input dimensions of
        a tf.function

        Parameters
        ----------
        states : 
            batch containing states
        actions : 
           batch containing actions 
        reward : 
           batch containing rewards to go 
        num_valid : 
           number of valid steps 

        Returns
        -------
         
            

        """
        with tf.GradientTape() as actor_tape:
            *_, mu, std, ddist, __, z = self.actor(states, training=True)
            dist = tfp.distributions.Normal(mu, std, allow_nan_stats=False)
            all_log_probs = dist.log_prob(actions)
            log_probs = tf.math.reduce_sum(all_log_probs, axis=-1)
            actor_loss = -log_probs * reward
            loss = tf.math.reduce_sum(actor_loss[:num_valid])
            # actor_loss = tf.math.reduce_sum(actor_loss)

        actor_grad = actor_tape.gradient(loss, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_weights))

        #         print(alpha)
        #         print(log_prob)
        return loss, mu[:num_valid]

    def train(self, num_epochs):
        """
        Training loop for to train with the REINFORCE agent

        Parameters
        ----------
        num_epochs : 
           number of training epochs 

        """
        actor_ckpt = tf.train.Checkpoint(optimizer=self.actor_optimizer,
                                         net=self.actor)
        manager = tf.train.CheckpointManager(actor_ckpt,
                                             './reinforce_ckpts',
                                             max_to_keep=100)
        self.is_test = False
        saved = False
        last_saved = 0
        for i in range(num_epochs):
            log_probs = np.zeros(1000, dtype=np.float32)
            rewards = np.zeros(1000, dtype=np.float32)
            states = np.zeros((1000, self.num_observations), dtype=np.float32)
            actions = np.zeros((1000, self.num_actions), dtype=np.float32)
            zs = np.zeros((1000, self.num_actions), dtype=np.float32)
            state, *_ = self.env.reset()
            num_runs = 0
            total_reward = 0
            for j in range(1000):
                next_state, reward, done, log_prob, action, truncated, z = self.take_step(
                    np.expand_dims(state, axis=0))
                total_reward += reward
                log_probs[j] = log_prob
                states[j] = state
                rewards[j] = reward
                actions[j] = action
                zs[j] = z
                num_runs = j
                if done == True:
                    if truncated == False:
                        rewards[j] = -5
                    break
                if truncated == True:
                    break
                state = next_state
            rewards_to_go = np.zeros_like(rewards)
            reward_to_go = 0.0
            for t in reversed(range(num_runs + 1)):
                reward_to_go = rewards[t] + self.gamma * reward_to_go
                rewards_to_go[t] = reward_to_go
            actor_loss, mu = self.update_model(
                tf.convert_to_tensor(states), tf.convert_to_tensor(zs),
                tf.convert_to_tensor(rewards_to_go), tf.Variable(num_runs + 1))
            mu = mu.numpy()
            wandb.log(
                {
                    'Actor loss': actor_loss / num_runs,
                    'Train Rewards': total_reward,
                    'Exploration Rate': self.exploration_rate
                },
                step=i)
            if total_reward > 100:
                if total_reward > last_saved:
                    # manager.save()
                    self.actor.save_weights(f'./checkpoints/{checkpoint_name}')
                last_saved = total_reward
                saved = True
            if i % 10 == 0:

                print('Evaluate')
                validation_reward = self.validate()
                wandb.log({'Validation Reward': validation_reward}, step=i)
                if validation_reward > 300:
                    self.actor.save_weights(f'./checkpoints/{checkpoint_name}')
                    saved = True
                if validation_reward > 990:
                    # manager.save()
                    self.actor.save_weights(
                        f'./checkpoints/{checkpoint_name}_val')
                    # env.close()
                    # return
                # manager.save()
            if i % 250 == 0:
                source = self.test(load=False)
                save_video(source, f'tmp/REINFORCE_4_{i}')
            self.exploration_rate = self.exploration_rate * self.exploration_rate_update
            self.env.close()
        if saved == False:

            self.actor.save_weights(f'./checkpoints/{checkpoint_name}')
            # manager.save()

        self.env.close()
        return

    def validate(self):
        """
        Validate current model

        Returns
        -------
        
           total reward from the validation run 

        """
        state, *_ = self.val_env.reset()
        self.is_test = True
        total_reward = 0
        for i in range(1000):
            # TODO Stuff
            state, reward, done = self.take_step(np.expand_dims(state, axis=0))
            total_reward += reward
            if done:
                break
        # self.val_env.close()
        self.is_test = False
        return total_reward

    def test(self, num=1000, env=None, load=True):
        """

        Parameters
        ----------
        num : 
          number of steps  
        env : 
           optinal new environment 
        load : 
           flag, if checkpoint should be loaded 

        Returns
        -------
        
            

        """
        if env is not None:
            self.val_env = env
        state, *_ = self.val_env.reset()
        # self.actor =  tf.keras.models.load_model('./reinforce_ckpts')

        # latest = tf.train.latest_checkpoint('./checkpoints/my_checkpoint')
        if load is True:
            self.actor.load_weights(f'./checkpoints/{checkpoint_name}')
        self.is_test = True
        total_reward = 0
        frames = []
        for i in range(1000):
            # TODO Stuff
            state, reward, done = self.take_step(np.expand_dims(state, axis=0))
            frames.append(self.val_env.render())
            total_reward += reward
            if done:
                break
        print(total_reward)
        # self.val_env.close()
        self.is_test = False
        return frames


def save_video(source, output_name):
    fps = 30
    out = cv2.VideoWriter(
        output_name + ".mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (source[0].shape[1], source[0].shape[0]),
    )
    for i in range(len(source)):
        out.write(source[i])
    out.release()


if __name__ == "__main__":
    # xvfb-run -a python basicAgentAsync.py
    wandb.init(
        # set the wandb project where this run will be logged
        project="Ant-REINFORCE-1",

        # track hyperparameters and run metadata with wandb.config
        config={
            "seed": 42,
            "ctrl_cost_weight": 0.4,
            "epochs": 30000,
            "exploration_rate": 0.12,
            "exploration_update": 0.999,
            "min_exploration": 0.06,
            "log_std_min": -3.5,
            "log_std_max": 2.5,
            "learning_rate": 1e-4,
            "gamma": 0.9,
            "gamma_alive": 0.999,
            "custom_reward": False
        })
    config = wandb.config
    tf.random.set_seed(42)
    env_name = 'Ant-v4'

    # env_name = 'InvertedPendulum-v4'

    val_env = gym.make(env_name, render_mode='rgb_array')
    env = gym.make(env_name,
                   max_episode_steps=1000,
                   ctrl_cost_weight=config.ctrl_cost_weight)
    agent = ReinforceAgent(env=env,
                           val_env=val_env,
                           exploration_rate=config.exploration_rate,
                           exploration_rate_update=config.exploration_update,
                           learning_rate=config.learning_rate,
                           gamma=config.gamma,
                           custom_reward=config.custom_reward,
                           gamma_survive=config.gamma_alive,
                           min_exploration=config.min_exploration,
                           log_std_min=config.log_std_min,
                           log_std_max=config.log_std_max)

    try:
        agent.train(config.epochs)
    except KeyboardInterrupt:
        agent.env.close()

    print('<===== TRAINED =====>')
    env = gym.make(env_name, render_mode="rgb_array")
    source = agent.test(1000, env)

    save_video(source, 'tmp/REINFORCE1')
