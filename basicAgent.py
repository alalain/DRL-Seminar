from model import Actor, CriticQ, CriticV
from replayBuffer import ReplayBuffer

import numpy as np
import time
import gymnasium as gym
import tensorflow as tf
from typing import Tuple


class BasicAgent:

    def __init__(
        self,
        env: gym.Env,
        buffer_size: int,
        batch_size: int,
        initial_random_steps: int,
        gamma: float = 0.99,
        tau: float = 5e-3,
    ):

        self.env = env

        self.initial_random_steps = initial_random_steps
        self.n_steps = tf.Variable(0)
        self.is_test = False
        self.batch_size = batch_size
        self.gamma = tf.Variable(gamma)
        self.policy_update_rate = tf.Variable(2)
        self.tau = tf.Variable(tau)

        num_observations = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        self.target_entropy = tf.constant(-np.prod(
            (num_actions, ), dtype=np.float32).item())  # heuristic
        self.log_alpha = tf.Variable(tf.zeros(1, dtype=tf.float32),
                                     name="log_alpha")
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

        self.replay_buffer = ReplayBuffer(num_observations, num_actions,
                                          buffer_size, batch_size)

        self.actor = Actor(num_actions, num_observations)
        self.q_function_a = CriticQ(num_observations)
        self.q_function_b = CriticQ(num_observations)
        self.v_function = CriticV(num_observations)
        self.v_target = CriticV(num_observations)
        self.v_target.set_weights(self.v_function.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        self.q_a_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        self.q_b_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        self.v_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

        self.q_a_loss_f = tf.keras.losses.MeanSquaredError()
        self.q_b_loss_f = tf.keras.losses.MeanSquaredError()
        self.vf_loss_f = tf.keras.losses.MeanSquaredError()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.n_steps < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = (self.actor(
                np.expand_dims(state,
                               axis=0), training=False)[0].numpy().squeeze(0))

        self.transition = [state, selected_action.squeeze()]

        return selected_action

    def take_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:

        next_state, reward, done, truncated, *_ = self.env.step(action)

        if not self.is_test:
            self.transition += [20 * reward, next_state, done]
            self.replay_buffer.store(*self.transition)
            done = done or truncated

        return next_state, 20 * reward, done

    @tf.function(reduce_retracing=True)
    def update_model(self, state, next_state, action, reward, done):
        with tf.GradientTape() as actor_tape, tf.GradientTape(
        ) as q_a_tape, tf.GradientTape() as q_b_tape, tf.GradientTape(
        ) as v_tape:
            with tf.GradientTape() as alphaTape:
                new_action, log_prob = self.actor(state)
                # alphaTape.watch(self.log_alpha)
                alpha_loss = tf.math.reduce_mean(
                    (-tf.exp(self.log_alpha) *
                     tf.stop_gradient(log_prob + self.target_entropy)))
                # tf.convert_to_tensor(log_prob) + self.target_entropy)))
            alpha_grad = alphaTape.gradient(alpha_loss, self.log_alpha)
            self.alpha_optimizer.apply_gradients(
                zip([alpha_grad], [self.log_alpha]))
            alpha = tf.exp(self.log_alpha)

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
                advantage = q_pred - tf.stop_gradient(v_pred)
                actor_loss = tf.math.reduce_mean(alpha * log_prob - advantage)

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

        #         print(alpha)
        #         print(log_prob)
        return actor_loss, q_a_loss, q_b_loss, vf_loss

    def train(self, num_iters):
        self.is_test = False
        state, *_ = self.env.reset()

        scores = []
        score = 0
        losses = []

        for i in range(1, num_iters):
            self.n_steps.assign_add(1)
            action = self.select_action(state)
            # print(action)
            state, reward, done = self.take_step(action)
            score += reward
            if done:
                state, *_ = self.env.reset()
                scores.append(score)
                score = 0

            if (len(self.replay_buffer) >= self.batch_size
                    and self.n_steps > self.initial_random_steps):
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

                loss = self.update_model(states, next_state, actions, rewards,
                                         dones)
                losses.append(loss)
            if i % 1000 == 0:
                print(f'Step {i}')
                if len(losses) > 0:
                    print(f'Actor: {loss[0]}')
                    print(f'Q A:   {loss[1]}')
                    print(f'Q B:   {loss[2]}')
                    print(f'Value: {loss[3]}')
                    print(f'Alpha: {tf.exp(self.log_alpha)}')
                    print(f'Reward:{np.mean(np.array(scores[-10:]))}')

        self.env.close()
        return scores

    def _target_soft_update(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.v_target.trainable_weights
        bases = self.v_function.trainable_weights
        for base, target in zip(bases, targets):
            target.assign(base * tau + target * (1 - tau))
            # weights.append(weight * tau + targets[i] * (1 - tau))
        # self.v_target.set_weights(weights)

    def test(self, num=1000, env=None):
        if env is not None:
            self.env = env
        state, *_ = self.env.reset()
        self.n_steps = 0
        self.is_test = True
        frames = []
        total_reward = 0
        for i in range(num):
            action = self.select_action(state)
            next_state, reward, done = self.take_step(action)
            frames.append(self.env.render())
            self.n_steps += 1
            total_reward += reward
            if done:
                break
        print(i)
        print(total_reward)
        self.env.close()
        self.n_steps = 0
        return frames

    def human_test(self, num=1000, env=None):
        if env is not None:
            self.env = env
        state, *_ = self.env.reset()
        self.n_steps = 0
        self.is_test = True
        frames = []
        total_reward = 0
        for i in range(num):
            action = self.select_action(state)
            next_state, reward, done = self.take_step(action)
            self.n_steps += 1
            total_reward += reward
            if done:
                print('gebrochen')
                break
        print(i)
        print(total_reward)
        self.n_steps = 0
        self.env.close()
        return frames


if __name__ == "__main__":
    # xvfb-run -a python basicAgent.py
    #     env = gym.make("Ant-v4", render_mode="rgb_array")
    #     env = gym.make("Ant-v5", render_mode="human")
    #     agent = BasicAgent(env, 100, 32, 2)
    #     source = agent.human_test()
    tf.random.set_seed(42)
    env_name = 'Ant-v4'
    # env_name = 'InvertedPendulum-v4'

    # env = gym.make(env_name, max_episode_steps=1000)
    env = gym.make(env_name,
                   max_episode_steps=1000,
                   ctrl_cost_weight=0.005,
                   healthy_reward=0.05)
    agent = BasicAgent(env=env,
                       buffer_size=500000,
                       batch_size=128,
                       initial_random_steps=250000)
    #     agent.human_test(4000)
    try:
        # scores = agent.train(150000)
        scores = agent.train(900000)
    except KeyboardInterrupt:
        scores = []
        agent.env.close()

    env = gym.make(env_name, render_mode="rgb_array")
    source = agent.test(1000, env)

    print(scores)
    import matplotlib.pyplot as plt
    plt.plot(scores)
    plt.savefig('tmp/reward.png')
    plt.show()

    print("trained")
    print("=" * 30)
    import cv2

    output_name = "tmp/test3Diag"
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
