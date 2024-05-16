from model import Actor, CriticQ, CriticV
from replayBuffer import ReplayBuffer

import numpy as np
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
        self.n_steps = 0
        self.is_test = False
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_update_rate = 2

        num_observations = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        self.target_entropy = tf.constant(
            -np.prod((num_actions,), dtype=np.float32).item()
        )  # heuristic
        self.log_alpha = tf.Variable(tf.zeros(1, dtype=tf.float32), name="log_alpha")
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        self.replay_buffer = ReplayBuffer(
            num_observations, num_actions, buffer_size, batch_size
        )

        self.actor = Actor(num_actions, num_observations)
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

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.n_steps < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = (
                self.actor(np.expand_dims(state, axis=0))[0].numpy().squeeze(0)
            )

        self.transition = [state, selected_action.squeeze()]

        return selected_action

    def take_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:

        next_state, reward, done, truncated, *_ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.replay_buffer.store(*self.transition)
            done = done or truncated

        return next_state, reward, done

    def update_model(self):
        samples = self.replay_buffer.sample_batch()
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = samples["acts"]
        reward = samples["rews"].reshape(-1, 1)
        done = samples["done"].reshape(-1, 1)
        with tf.GradientTape() as actor_tape, tf.GradientTape() as q_a_tape, tf.GradientTape() as q_b_tape, tf.GradientTape() as v_tape:
            with tf.GradientTape() as alphaTape:
                new_action, log_prob = self.actor(state)
                alphaTape.watch(self.log_alpha)
                alpha_loss = tf.math.reduce_mean(
                    (
                        -tf.exp(self.log_alpha)
                        * tf.stop_gradient(
                            tf.convert_to_tensor(log_prob) + self.target_entropy
                        )
                    )
                )
            alpha_grad = alphaTape.gradient(alpha_loss, self.log_alpha)
            self.alpha_optimizer.apply_gradients(zip([alpha_grad], [self.log_alpha]))
            alpha = tf.exp(self.log_alpha)

            mask = 1 - done
            q_a_pred = self.q_function_a(state, action)
            q_b_pred = self.q_function_b(state, action)
            v_target = self.v_target(next_state)
            q_target = reward + self.gamma * v_target * done
            q_a_loss = self.q_a_loss_f(q_a_pred, tf.stop_gradient(q_target))
            q_b_loss = self.q_b_loss_f(q_b_pred, tf.stop_gradient(q_target))

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
            actor_grad = actor_tape.gradient(actor_loss, self.actor.trainable_weights)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_weights)
            )
        #             self._target_soft_update()

        q_a_grad = q_a_tape.gradient(q_a_loss, self.q_function_a.trainable_weights)
        self.q_a_optimizer.apply_gradients(
            zip(q_a_grad, self.q_function_a.trainable_weights)
        )

        q_b_grad = q_b_tape.gradient(q_b_loss, self.q_function_b.trainable_weights)
        self.q_b_optimizer.apply_gradients(
            zip(q_b_grad, self.q_function_b.trainable_weights)
        )

        v_grad = v_tape.gradient(vf_loss, self.v_function.trainable_weights)
        self.v_optimizer.apply_gradients(zip(v_grad, self.v_function.trainable_weights))

#         print(alpha)
#         print(log_prob)
        return actor_loss.numpy(), q_a_loss.numpy(), q_b_loss.numpy(), vf_loss.numpy()

    def train(self, num_iters):
        self.is_test = False
        state, *_ = self.env.reset()

        scores = []
        score = 0
        losses = []

        for self.n_steps in range(1, num_iters):
            action = self.select_action(state)
            state, reward, done = self.take_step(action)
            score += reward
            if done:
                state, *_ = self.env.reset()
                scores.append(score)
                score = 0

            if (
                len(self.replay_buffer) >= self.batch_size
                and self.n_steps > self.initial_random_steps
            ):
                loss = self.update_model()
                losses.append(loss)
            if self.n_steps % 100 == 0:
                print(f'Step {self.n_steps}')
                if len(losses) > 0:
                    print(f'Actor: {loss[0]}')
                    print(f'Q A:   {loss[1]}')
                    print(f'Q B:   {loss[2]}')
                    print(f'Value: {loss[3]}')

        self.env.close()
        return scores

    def _target_soft_update(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.v_target.weights
        for i, weight in enumerate(self.v_function.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.v_target.set_weights(weights)

    def test(self):
        state, *_ = self.env.reset()
        self.n_steps = 0
        self.is_test = True
        frames = []
        for i in range(100):
            action = self.select_action(state)
            next_state, reward, done = self.take_step(action)
            frames.append(self.env.render())
            self.n_steps += 1
            if done:
                break
        print(i)
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

    env = gym.make("Ant-v4", max_episode_steps=1000, terminate_when_unhealthy=True)
    agent = BasicAgent(env, 20000, 128, 10000)
#     agent.human_test(4000)
    scores = agent.train(15500)
    env = gym.make("Ant-v4", render_mode="human")
    agent.human_test(4000, env=env)


    print(scores)
    import matplotlib.pyplot as plt
    plt.plot(scores)
    plt.show()

    print("trained")
    print("=" * 30)
    import cv2

    output_name = "tmp/test"
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
