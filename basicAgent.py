from model import Actor, CriticQ, CriticV
from replayBuffer import ReplayBuffer

import numpy as np
import gymnasium as gym
from typing import Tuple


class BasicAgent():

    def __init__(self, env: gym.Env, buffer_size: int, batch_size: int,
                 initial_random_steps: int):

        self.env = env

        self.initial_random_steps = initial_random_steps
        self.n_steps = 0
        self.is_test = False

        num_observations = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        self.replay_buffer = ReplayBuffer(num_observations, buffer_size,
                                          batch_size)

        self.actor = Actor(num_actions, num_observations)
        self.q_function_a = CriticQ(num_observations)
        self.q_function_b = CriticQ(num_observations)
        self.v_function = CriticV(num_observations)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.n_steps < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.actor(np.expand_dims(
                state, axis=0))[0].numpy().squeeze(0)

        self.transition = [state, selected_action]

        return selected_action

    def take_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:

        next_state, reward, done, *_ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def train(self, num_iters):
        self.is_test = False
        state, *_ = self.env.reset()

        scores = []
        score = 0

        for self.n_steps in range(1, num_iters):
            action = self.select_action(state)
            state, reward, done = self.take_step(action)
            score += reward
            if done:
                state, *_ = self.env.reset()
                scores.append(score)
                score = 0


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
        self.n_steps = 0
        return frames


if __name__ == '__main__':
    # xvfb-run -a python basicAgent.py
    env = gym.make('Ant-v4', render_mode="rgb_array")
    agent = BasicAgent(env, 100, 32, 2)
    source = agent.test()

    import cv2
    output_name = 'tmp/test'
    fps = 30
    out = cv2.VideoWriter(output_name + '.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (source[0].shape[1], source[0].shape[0]))
    for i in range(len(source)):
        out.write(source[i])
    out.release()
