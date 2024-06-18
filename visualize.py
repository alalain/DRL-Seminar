import tensorflow as tf
import gymnasium as gym
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

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

stoc_envs = gym.make_vec(
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

actor = Actor(8, 27, -2.5, 3.5)
actor(np.random.randn(1, 27))
# actor.load_weights('./tf_ckpts_2/ckpt-32.data-00000-of-00001.ckpt')

actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
actor_ckpt = tf.train.Checkpoint(optimizer=actor_optimizer, net=actor)
ckpt_path = './tf_ckpts_4/ckpt-'
# manager = tf.train.CheckpointManager(actor_ckpt,
# './tf_ckpts_3',
# max_to_keep=100)
# manager.restore_or_initialize()
mean_rewards = []
stoc_mean_rewards = []
for i in range(1, 100):
    # mean_rewards.append(i)
    # continue
    try:
        actor_ckpt.restore(ckpt_path + f'{i}')
    except:
        print('End')
        print(f'Loaded {i} checkpoints')
        break
# exit()
# manager.restore(manager.latest_checkpoint)
# actor.load_weights('./tf_ckpts_2/ckpt-32')

    state, *_ = val_env.reset()
    states, *_ = envs.reset()
    stoc_states, *_ = stoc_envs.reset()
    frames = []
    frames.append(val_env.render())
    done_mask = np.ones(10, dtype=np.bool_)
    stoc_done_mask = np.ones(10, dtype=np.bool_)
    total_rewards = np.zeros(10)
    stoc_total_rewards = np.zeros(10)
    total_reward = 0
    single_run = 1
    for i in range(1000):

        # action, log_prob, mu,std = actor(np.expand_dims(state, axis=0))
        # action = tf.math.tanh(mu)
        # action = action.numpy().squeeze()
        # next_state, reward, done, trunc, info = val_env.step(action)
        # frames.append(val_env.render())
        # state = next_state
        # single_run = 1-done
        # total_reward += reward * single_run

        actions, log_probs, mus, stds = actor(states)
        actions = tf.math.tanh(mus)
        actions = actions.numpy()
        next_states, rewards, dones, truncs, infos = envs.step(actions)

        done_mask = done_mask & np.invert(dones)
        total_rewards += rewards * done_mask
        states = next_states

        # stoc_actions, log_probs, stoc_mus, stds = actor(stoc_states)
        # actions = stoc_actions.numpy()
        # stoc_next_states, stoc_rewards, stoc_dones, stoc_truncs, stoc_infos = stoc_envs.step(
        #     actions)
        #
        # stoc_done_mask = stoc_done_mask & np.invert(stoc_dones)
        # stoc_total_rewards += stoc_rewards * stoc_done_mask
        # stoc_states = stoc_next_states

    # print(total_reward)
    # save_video(frames, 'tmp/wadafug_6')
    print(np.mean(total_rewards))
    mean_rewards.append((np.mean(total_rewards)))
    # print('-------')
    # print(np.mean(stoc_total_rewards))
    # stoc_mean_rewards.append((np.mean(stoc_total_rewards)))
    # print('###########################')
fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
ax.plot(np.arange(1, len(mean_rewards) + 1) * 50000, mean_rewards, label='Deterministic Action')
# ax.plot(np.arange(1, len(mean_rewards) + 1) * 50000, stoc_mean_rewards, label='Sampled Actions')
ax.set_ylabel('Total Reward')
ax.set_xlabel('Epoch')
ax.grid()
# ax.legend(loc='best')
plt.savefig("./doc/images/val_adv.pdf")

# xvfb-run -a python visualize.py
