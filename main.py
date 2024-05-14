    import numpy as np
import tensorflow as tf
import gymnasium as gym
import replayBuffer as rb

def main():
    env = gym.make('Ant-v4', render_mode="human")
#     env = gym.make('Ant-v5')
    observation, _ = env.reset()
    for i in range(1000):
        action = np.random.random(8) * 2 - 1
    #     _ = env.step(action)
        _ = env.step(env.action_space.sample())
#         env.render()
    env.render()
    env.close()

if __name__ == '__main__':
    main()
