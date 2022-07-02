import numpy as np
import h5py
import gym
from tqdm import tqdm
import os
import argparse

from dexterity.utils.dm2gym import GymEnv
from dexterity import manipulation

def pretty(d, indent=0):
  for key, value in d.items():
    print('\t' * indent + str(key))
    if isinstance(value, dict):
      pretty(value, indent + 1)
    else:
      print('\t' * (indent + 1) + str(value))

def get_keys(h5file):
  keys = []
  def visitor(name, item):
    if isinstance(item, h5py.Dataset):
      keys.append(name)
  h5file.visititems(visitor)
  return keys

def save_hdf5(observations, actions, rewards, terminals, fname):
  with h5py.File(fname, 'w') as f:
    f.create_dataset('observations', data=observations)
    f.create_dataset('actions', data=actions)
    f.create_dataset('rewards', data=rewards)
    f.create_dataset('terminals', data=terminals)

def main(args):
  env = GymEnv(domain_name="roller", task_name="state_dense")
  dataset_file = h5py.File(
      f'logs/{args.input}.hdf5', 'r')
  dataset = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
  dataset_file.close()
  dataset['terminals'][-1] = 1
  # keys: terminals, rewards, actions, observations
  term_list, rew_list, act_list, obs_list = [], [], [], []
  end_point = np.where(dataset['terminals'].flatten())[0]
  end_point = np.concatenate(([-1], end_point))
  for i in tqdm(range(len(end_point))):
    start = end_point[i-1] + 1
    end = end_point[i]
    traj_length = end - start
    for goal_idx in range(start, end+1):
      ag = gym.spaces.utils.unflatten(env.observation_space,(dataset['observations'][goal_idx]))['prop/orientation']
      for trans_idx in range(start, goal_idx+1):
        old_obs = dataset['observations'][trans_idx]
        obs_dict = gym.spaces.utils.unflatten(env.observation_space, old_obs) 
        obs_dict['target_prop/orientation'] = ag
        obs = gym.spaces.utils.flatten(env.observation_space, obs_dict)
        obs_list.append(obs)
        term = [float(trans_idx == goal_idx)]
        term_list.append(term)
        rew_list.append(dataset['rewards'][trans_idx])
        act_list.append(dataset['actions'][trans_idx])
  observations = np.array(obs_list, dtype=np.float32)
  actions = np.array(act_list, dtype=np.float32)
  rewards = np.array(rew_list, dtype=np.float32)
  terminals = np.array(term_list, dtype=np.float32)

  buffer_path = os.path.join('logs', f'{args.output}.hdf5')
  save_hdf5(observations, actions, rewards, terminals, buffer_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', type=str, default='origin')
  parser.add_argument('--output', type=str, default='relabel')
  args = parser.parse_args()
  main(args)