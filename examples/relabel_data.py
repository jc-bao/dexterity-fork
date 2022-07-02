import numpy as np
import gym
import argparse
import os
import h5py
from tqdm import tqdm

from dexterity.utils.dm2gym import GymEnv
from dexterity import manipulation

def save_hdf5(observations, actions, rewards, terminals, fname):
  with h5py.File(fname, 'w') as f:
    f.create_dataset('observations', data=observations)
    f.create_dataset('actions', data=actions)
    f.create_dataset('rewards', data=rewards)
    f.create_dataset('terminals', data=terminals)


def save_buffer(buffer, logdir):
  observations = []
  actions = []
  rewards = []
  terminals = []
  for i in range(len(buffer)):
    observations.append(buffer[i][0])
    actions.append(buffer[i][1])
    rewards.append(buffer[i][2])
    terminals.append(buffer[i][3])

  observations = np.array(observations, dtype=np.float32)
  actions = np.array(actions, dtype=np.float32)
  rewards = np.array(rewards, dtype=np.float32)
  terminals = np.array(terminals, dtype=np.float32)

  buffer_path = os.path.join(logdir, 'buffer.hdf5')
  save_hdf5(observations, actions, rewards, terminals, buffer_path)


def collect(env, policy, logdir, final_step):
  buffer = []

  step = 0
  pbar = tqdm(total=final_step)
  while step < final_step:
    obs_t = env.reset()
    ter_t = False
    rew_t = 0.0
    while step < final_step and not ter_t:
      obs_dict = gym.spaces.utils.unflatten(env.unwrapped.observation_space, obs_t)
      act_t = policy(obs_dict)
      buffer.append([obs_t, act_t, [rew_t], [ter_t]])
      obs_t, rew_t, ter_t, _ = env.step(act_t)
      step += 1
      pbar.update(1)
    if ter_t:
      buffer.append([obs_t, np.zeros_like(act_t), [rew_t], [ter_t]])

  save_buffer(buffer[:final_step], logdir)
  print('Collected data has been saved.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--final-step', type=int, default=1000000)
  parser.add_argument('--policy', type=str, default='handcrafted')
  args = parser.parse_args()

  env = GymEnv(domain_name="roller", task_name="state_dense")
  env = gym.wrappers.FlattenObservation(env)

  observation_size = env.observation_space.shape[0]
  action_size = env.action_space.shape[0]

  collect(env, manipulation.ezpolicy, 'logs', args.final_step)