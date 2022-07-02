import numpy as np
from gym.spaces.utils import unflatten

def ezpolicy(obs):
  # current state
  joint_pos = obs['roller_hand/joint_positions']
  pitch_r, pitch_l = joint_pos[1], joint_pos[4]
  obj_quat = obs['prop/orientation']
  current_roll = np.arccos(obj_quat[0])
  if obj_quat[3] < 0:
    current_roll = -current_roll
  obj_y, obj_z = obs['prop/position'][1:]
  obj_z -= 0.05
  # target state
  goal_orn = obs['goal_state']
  yz = goal_orn[2:4]
  target_pitch = np.arctan(yz[0] / yz[1])
  target_roll = np.arccos(goal_orn[0])
  if goal_orn[3] < 0:
    target_roll = -target_roll
  obj_disp = -obj_y * np.cos(target_pitch) + obj_z * np.sin(target_pitch)
  # preprocess angle
  if current_roll > target_roll + np.pi:
    current_roll -= 2 * np.pi
  elif current_roll < target_roll - np.pi:
    current_roll += 2 * np.pi
  assert abs(current_roll - target_roll) <= np.pi
  # action
  action = np.zeros(4)
  pitch_err_l = target_pitch - pitch_r
  pitch_err_r = target_pitch - pitch_l
  if (abs(pitch_err_l) > 0.01) and (abs(pitch_err_r) > 0.01):
    action[0] = target_pitch
    action[2] = target_pitch
  else:
    action[0] = target_pitch
    action[2] = target_pitch
    if current_roll < target_roll - 0.01:
      action[1] = 1
    elif current_roll > target_roll + 0.01:
      action[1] = -1
    action[3] = action[1]
    action[1] += obj_disp * 100
    action[3] -= obj_disp * 100
  return action