from model import Actor, CriticQ, CriticV
from replayBuffer import ReplayBuffer

import numpy as np
import gymnasium as gym


class BasicAgent():


    def __init__(self,
             env: gym.Env,
             buffer_size: int,
             batch_size: int,
             initial_random_steps: int
        ):

        self.env = env

        self.initial_random_steps= initial_random_steps
        self.n_steps = 0
        self.is_test = False

        num_observations = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        self.replay_buffer = ReplayBuffer(num_observations, buffer_size, batch_size)

        self.actor = Actor(num_actions, num_observations)
        self.q_function = CriticQ(num_observations)
        self.v_function = CriticV(num_observations)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.n_steps < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.actor(
                np.expand_dims(state, axis=0)
            )[0].numpy().squeeze(0)

        self.transition = [state, selected_action]

        return selected_action


    def test(self):
        state, *_ = self.env.reset()
        self.n_steps = 0
        self.is_test = True
        for i in range(100):
            action = self.select_action(state)
            next_state, reward, done, *_ = self.env.step(action)
            self.n_steps += 1
            if done:
                break
        print(i)
        self.n_steps = 0

if __name__ == '__main__':
    env = gym.make('Ant-v4', render_mode="human")
    agent = BasicAgent(env, 100, 32, 2)
    agent.test()
