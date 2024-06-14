import tensorflow as tf
import gymnasium as gym
import numpy as np
import cv2 as cv2

from model import Actor


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

envs = gym.make_vec(
        "Ant-v4",
        num_envs=10,
        healthy_reward=1,
        # max_episode_steps=1000,
        # use_contact_forces=True,
        ctrl_cost_weight=0.5,
        vectorization_mode='async')

val_env = gym.make("Ant-v4",
               healthy_reward=1,
               ctrl_cost_weight=0.5,
               render_mode='rgb_array')

actor = Actor(8, 27,-2.5, 3.5)
actor(np.random.randn(1,27))
# actor.load_weights('./tf_ckpts_2/ckpt-32.data-00000-of-00001.ckpt')

actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
actor_ckpt = tf.train.Checkpoint(optimizer=actor_optimizer,
                                 net=actor)
manager = tf.train.CheckpointManager(actor_ckpt,
                                             './tf_ckpts_3',
                                             max_to_keep=100)

manager.restore_or_initialize()
# manager.restore(manager.latest_checkpoint)
# actor.load_weights('./tf_ckpts_2/ckpt-32')

state, *_ = val_env.reset()
states, *_ = envs.reset()
frames = []
frames.append(val_env.render())
done_mask = np.ones(10, dtype=np.bool_)
total_rewards = np.zeros(10)
total_reward = 0
single_run = 1
for i in range(1000):

    action, log_prob, mu,std = actor(np.expand_dims(state, axis=0))
    action = tf.math.tanh(mu)
    # actions, *mus = actor(states)
    action = action.numpy().squeeze()
    # actions = actions.numpy()
    next_state, reward, done, trunc, info = val_env.step(action) 
    frames.append(val_env.render())
    # next_states, rewards, dones, truncs, infos = envs.step(actions)
    
    # import ipdb; ipdb.set_trace()
    # rewards = infos['reward_survive'] * infos[
                # '_reward_survive'] + infos['reward_forward'] * 5 * infos[
                    # '_reward_forward'] + infos['reward_ctrl'] * infos[
                        # '_reward_ctrl']

    single_run = 1-done
    total_reward += reward * single_run

    # done_mask = done_mask & np.invert(dones)
    # total_rewards += rewards * done_mask
    state = next_state
    # states = next_states

print(total_reward)
save_video(frames, 'tmp/wadafug_6')
# print(total_rewards)

