from model import Actor, CriticQ, CriticV
from replayBuffer import ReplayBuffer

import numpy as np
import time
import gymnasium as gym
import tensorflow as tf
from typing import Tuple
import cv2
import wandb


class BasicAgent:

    def __init__(self,
                 env: gym.Env,
                 val_env: gym.Env,
                 buffer_size: int,
                 batch_size: int,
                 initial_random_steps: int,
                 gamma: float = 0.99,
                 tau: float = 5e-3,
                 reward_scale: float = 5,
                 gamma_survive: float = 0.99995,
                 num_envs: int = 10,
                 log_std_min: float = -3.5,
                 log_std_max: float = 2.5):
        """
        Parameters
        ----------
        env
            The Environment the agent will train in
        val_env
            Environment to for evaluation purposes
        buffer_size
            Size of the Replay Buffer used by the Agent
        batch_size
            Batch size used in each training step
        initial_random_steps
            Number of random steps performed at the beginning of training
        gamma
            Gamma for exponential decaying of the reward
        tau
            Smoothing factor for the soft updtae of the value network
        reward_scale
            Scale applied to the reward given by the environment
        gamma_survive
            Decay factor for the reward for surviving
        num_envs
            Number of parallel environments
        log_std_min
            min std of actor distribution
        log_std_max
            max std of actor distribution
        """

        self.env = env
        self.val_env = val_env

        self.initial_random_steps = initial_random_steps
        self.n_steps = tf.Variable(0)
        self.is_test = False
        self.batch_size = batch_size
        self.gamma = tf.Variable(gamma)
        self.policy_update_rate = tf.Variable(2)
        self.tau = tf.Variable(tau)

        self.gamma_survive = gamma_survive
        self.current_gamma_survive = gamma_survive
        self.reward_scale = reward_scale

        self.num_envs = num_envs

        num_observations = env.observation_space.shape[-1]
        num_actions = env.action_space.shape[-1]

        self.target_entropy = tf.constant(-np.prod(
            (num_actions, ), dtype=np.float32).item())  # heuristic
        # self.target_entropy *= 0.69
        self.log_alpha = tf.Variable(tf.zeros(1, dtype=tf.float32),
                                     name="log_alpha")
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-6)

        self.replay_buffer = ReplayBuffer(num_observations, num_actions,
                                          buffer_size, batch_size)

        self.actor = Actor(num_actions, num_observations, log_std_min,
                           log_std_max)
        self.q_function_a = CriticQ(num_observations)
        self.q_function_b = CriticQ(num_observations)
        self.v_function = CriticV(num_observations)
        self.v_target = CriticV(num_observations)
        self.v_target.set_weights(self.v_function.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.q_a_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.q_b_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.v_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        self.q_a_loss_f = tf.keras.losses.MeanSquaredError()
        self.q_b_loss_f = tf.keras.losses.MeanSquaredError()
        self.vf_loss_f = tf.keras.losses.MeanSquaredError()

    @tf.function()
    def run_model(self, state):
        """
        Tf function to speed up action selection

        Parameters
        ----------
        state : 
            state for which the next action shold be calculated

        Returns
        -------
            selected action and the mus from the underlying distribution
        """
        selected_action, log_prob, mu, std = self.actor(state, training=False)
        return selected_action, mu

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Selects the next action, the agent will perform. This function also handles
        the initial random steps

        Parameters
        ----------
        state : np.ndarray
            Current state of in the environments
            

        Returns
        -------
        np.ndarray
            selected action
            

        """
        # if initial random action should be conducted
        if self.n_steps < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            # selected_action = (self.actor(state, training=False)[0].numpy())
            selected_action, mu = self.run_model(tf.convert_to_tensor(state))
            # if self.is_test is True:
                # selected_action = tf.keras.activations.tanh(mu)
                # pass
            selected_action = selected_action.numpy()

        self.transition = [state, selected_action]

        return selected_action

    def take_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:

        """
        Takes a step in the environment based on the provided action.
        Also handles the reward and pushing the new data to the replay buffer

        Parameters
        ----------
        action : np.ndarray
            action to perform
            

        Returns
        -------
        Tuple[np.ndarray, float, bool]
            Contains: the next state, current reward and if the done flags from the environment
            

        """
        if not self.is_test:
            next_state, reward, done, truncated, info = self.env.step(action)
            # custom_reward = info['reward_survive'] * info[
                # '_reward_survive'] + info['reward_forward'] * 5 * info[
                    # '_reward_forward'] + info['reward_ctrl'] * info[
                        # '_reward_ctrl']
            self.current_gamma_survive *= self.gamma_survive
            # reward = custom_reward
            self.transition += [
                reward * self.reward_scale, next_state, done
            ]
            self.replay_buffer.store(*self.transition)

            done = done | truncated
            reward = reward

            return next_state, reward, done

        next_state, reward, done, truncated, info = self.val_env.step(action)
        return next_state, reward, done

    # @tf.function(reduce_retracing=True)
    @tf.function()
    def update_model(self, state, next_state, action, reward, done):
        """Perform a training step on the SAC networks 

        Parameters
        ----------
        state : 
            batch containing the states
            
        next_state : 
            batch containing the next states
            
        action : 
            batch containing the performed actions
            
        reward : 
            batch containing the rewards
            
        done : 
            batch containing the done flags
            

        Returns
        -------
        actor, q-function, value-function and alpha parameter loss
        
            

        """
        with tf.GradientTape() as actor_tape, tf.GradientTape(
        ) as q_a_tape, tf.GradientTape() as q_b_tape, tf.GradientTape(
        ) as v_tape:
            with tf.GradientTape() as alphaTape:
                new_action, log_prob, *_ = self.actor(state)
                # alphaTape.watch(self.log_alpha)
                alpha_loss = tf.math.reduce_mean(
                    -tf.exp(self.log_alpha) *
                    (tf.stop_gradient(log_prob + self.target_entropy)))
                # tf.convert_to_tensor(log_prob) + self.target_entropy)))
            alpha_grad = alphaTape.gradient(alpha_loss, self.log_alpha)
            self.alpha_optimizer.apply_gradients(
                zip([alpha_grad], [self.log_alpha]))
            alpha = tf.exp(self.log_alpha)
            # alpha = self.log_alpha

            mask = 1 - done
            q_a_pred = self.q_function_a(state, action)
            q_b_pred = self.q_function_b(state, action)
            v_target = self.v_target(next_state)
            q_target = reward + self.gamma * v_target * mask
            q_a_loss = self.q_a_loss_f(q_a_pred, tf.stop_gradient(q_target))
            q_b_loss = self.q_b_loss_f(q_b_pred, tf.stop_gradient(q_target))

            # tf.print(action.shape)
            # tf.print(new_action.shape)
            v_pred = self.v_function(state)
            q_pred = tf.math.minimum(
                self.q_function_a(state, new_action),
                self.q_function_b(state, new_action),
            )
            v_target = q_pred - alpha * log_prob
            vf_loss = self.vf_loss_f(v_pred, tf.stop_gradient(v_target))

            actor_loss = tf.zeros(1)
            if self.n_steps % self.policy_update_rate == 0:
                # advantage = q_pred - tf.stop_gradient(v_pred)
                # actor_loss = tf.math.reduce_mean(alpha * log_prob - advantage)
                actor_loss = tf.math.reduce_mean(alpha * log_prob - q_pred)

        if self.n_steps % self.policy_update_rate == 0:
            actor_grad = actor_tape.gradient(actor_loss,
                                             self.actor.trainable_weights)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_weights))
            self._target_soft_update()

        q_a_grad = q_a_tape.gradient(q_a_loss,
                                     self.q_function_a.trainable_weights)
        self.q_a_optimizer.apply_gradients(
            zip(q_a_grad, self.q_function_a.trainable_weights))

        q_b_grad = q_b_tape.gradient(q_b_loss,
                                     self.q_function_b.trainable_weights)
        self.q_b_optimizer.apply_gradients(
            zip(q_b_grad, self.q_function_b.trainable_weights))

        v_grad = v_tape.gradient(vf_loss, self.v_function.trainable_weights)
        self.v_optimizer.apply_gradients(
            zip(v_grad, self.v_function.trainable_weights))

        return actor_loss, q_a_loss, q_b_loss, vf_loss

    def train(self, num_iters):
        """
        Training functon to train with the Agent

        Parameters
        ----------
        num_iters : 
            Number of training steps to perform
            

        Returns
        -------
        Tuple(list, list)
            lists containing the trainign scores and validation scores

        """
        actor_ckpt = tf.train.Checkpoint(optimizer=self.actor_optimizer,
                                         net=self.actor)
        manager = tf.train.CheckpointManager(actor_ckpt,
                                             './tf_ckpts_5',
                                             max_to_keep=100)
        self.is_test = False
        state, *_ = self.env.reset()

        scores = []
        val_scores = []
        score = np.zeros(self.num_envs)
        losses = []
        save_num = 0
        while self.n_steps < num_iters:
            action = self.select_action(state)
            state, reward, done = self.take_step(action)
            score += reward
            for jj, (r, d, s) in enumerate(zip(reward, done, score)):
                if d == True:
                    scores.append(s)
                    score[jj] = 0


            if (len(self.replay_buffer) >= self.batch_size
                    and self.n_steps > self.initial_random_steps):
                for j in range(self.num_envs):
                    # t3 = time.time()
                    self.n_steps.assign_add(1)
                    samples = self.replay_buffer.sample_batch()
                    states = samples["obs"]
                    next_state = samples["next_obs"]
                    actions = samples["acts"]
                    rewards = samples["rews"].reshape(-1, 1)
                    dones = samples["done"].reshape(-1, 1)
                    states = tf.convert_to_tensor(states)
                    next_state = tf.convert_to_tensor(next_state)
                    actions = tf.convert_to_tensor(actions)
                    rewards = tf.convert_to_tensor(rewards)
                    dones = tf.convert_to_tensor(dones)

                    loss = self.update_model(states, next_state, actions,
                                             rewards, dones)

                    losses.append(loss)
            else:
                self.n_steps.assign_add(self.num_envs)
            if self.n_steps % 500 == 0:
                print(f'Step {self.n_steps.numpy()}')
                if len(losses) > 0:
                    wandb.log(
                        {
                            'Actor loss': loss[0],
                            'Q_a loss': loss[1],
                            'Q_b loss': loss[2],
                            'Vf loss': loss[3],
                            'Alpha': tf.exp(self.log_alpha),
                            'Log Alpha': self.log_alpha,
                            'Train Rewards': np.mean(np.array(scores[-5:]))
                        },
                        step=self.n_steps.numpy())
                else:
                    wandb.log(
                        {'Train Rewards': np.mean(np.array(scores[-5:]))},
                        step=self.n_steps.numpy())

            if self.n_steps % 50000 == 0 and self.n_steps > self.initial_random_steps:
                validation = self.validate()
                val_scores.append(validation)
                if (len(self.replay_buffer) >= self.batch_size
                        and self.n_steps > self.initial_random_steps):
                    manager.save()
                # print(f'<====== Validation:   {validation}')
                wandb.log({"validation Reward": validation})
            if self.n_steps % 100000 == 0 and self.n_steps > self.initial_random_steps:
                source = self.test()
                save_video(source, f'tmp/SAC_Q_LOSS{save_num}')
                save_num += 1

        self.env.close()
        return scores, val_scores

    def _target_soft_update(self, tau=None):
        tau = self.tau
        weights = []
        targets = self.v_target.trainable_weights
        bases = self.v_function.trainable_weights
        for base, target in zip(bases, targets):
            target.assign(base * tau + target * (1 - tau))

    def validate(self):
        """
        Run a validation run on the validation environment

        Returns
        -------
        total reward achieved by the validation run

        """
        state, *_ = self.val_env.reset()
        self.is_test = True
        total_reward = 0
        for i in range(1000):
            action = self.select_action(np.expand_dims(state,
                                                       axis=0)).squeeze()
            next_state, reward, done = self.take_step(action)
            total_reward += reward
            if done:
                break
            state = next_state
        self.is_test = False
        return total_reward

    def test(self, num=1000, env=None):
        """
        Runs a validation/test run and renders the environment each step

        Parameters
        ----------
        num : 
            number of steps
            
        env : 
            optional: Environment to run the test in
            

        Returns
        -------
        list of rendered images

        """
        if env is not None:
            self.val_env = env
        state, *_ = self.val_env.reset()
        self.is_test = True
        frames = []
        total_reward = 0
        for i in range(num):
            action = self.select_action(np.expand_dims(state,
                                                       axis=0)).squeeze()
            next_state, reward, done = self.take_step(action)
            frames.append(self.val_env.render())
            total_reward += reward
            if done:
                break
            state = next_state
        print(i)

        print(total_reward)
        # self.val_env.close()
        self.is_test = False
        return frames

    def random_test(self, num=1000, env=None):
        """
        Runs the environment with random steps

        Parameters
        ----------
        num : 
            max number of steps
            
        env : 
            environment to run in
            

        Returns
        -------

        """
        if env is not None:
            self.val_env = env
        state, *_ = self.val_env.reset()
        self.is_test = True
        frames = []
        total_reward = 0
        for i in range(num):
            action = self.val_env.action_space.sample()
            next_state, reward, done = self.take_step(action)
            frames.append(self.val_env.render())
            total_reward += reward
            if done:
                break
            state = next_state
        print(i)
        print(total_reward)
        # self.val_env.close()
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
        project="Ant-SAC-1",

        # track hyperparameters and run metadata with wandb.config
        config={
            "seed": 420,
            "ctrl_cost_weight": 0.5,
            "random_steps": 200000,
            "buffer_size": 1000000,
            "batch_size": 256,
            "training_steps": 4000000,
            "log_std_min": -2.5,
            "log_std_max": 3.5,
            "reward_scale": 5,
            "alpha_lr": 3e-6,
            "network_lr": 3e-4,
            "Advantage Loss": False
        })
    config = wandb.config
    tf.random.set_seed(config.seed)
    env_name = 'Ant-v4'

    # env_name = 'InvertedPendulum-v4'

    # set up environements
    envs = gym.make_vec(
        "Ant-v4",
        num_envs=10,
        healthy_reward=1,
        ctrl_cost_weight=config.ctrl_cost_weight,
        vectorization_mode='async')
    env = gym.make(env_name,
                   max_episode_steps=1000,
                   healthy_reward=1,
                   ctrl_cost_weight=config.ctrl_cost_weight)
    val_env = gym.make(env_name,
                       healthy_reward=1,
                       ctrl_cost_weight=config.ctrl_cost_weight,
                       render_mode='rgb_array')
    # setup agent
    agent = BasicAgent(env=envs,
                       val_env=val_env,
                       buffer_size=config.buffer_size,
                       batch_size=config.batch_size,
                       reward_scale=config.reward_scale,
                       initial_random_steps=config.random_steps,
                       log_std_min=config.log_std_min,
                       log_std_max=config.log_std_max)

    # initial random run for testign purposes
    source = agent.random_test()
    save_video(source, 'tmp/random')
    # train agent, allowing to interrupt with keyboard interrupt
    try:
        scores, val_scores = agent.train(config.training_steps)
    except KeyboardInterrupt:
        scores = []
        val_scores = []
        agent.env.close()

    print('<===== TRAINED =====>')
    env = gym.make(env_name, render_mode="rgb_array")
    source = agent.test(1000, env)

