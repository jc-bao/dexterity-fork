import numpy as np
from gym.spaces.utils import unflatten
from scipy.spatial.transform import Rotation as R

PITCH_POS2VEL = 20
ROLL_POS2VEL = 20

def ezpolicy(obs, observation_space=None):
  if isinstance(obs, np.ndarray):
    obs = unflatten(observation_space, obs) 
  # ===parse observation=== 
  goal_orn = R.from_quat([*obs['goal_state'][1:], obs['goal_state'][0]])
  obj_orn = R.from_quat([*obs['prop/orientation'][1:], obs['prop/orientation'][0]])
  robot_orn = R.from_euler('z', obs['roller_hand/joint_positions'][0])
  diff_orn = goal_orn * obj_orn.inv()
  pitch = -obs['roller_hand/joint_positions'][2]%(2*np.pi)
  # ===calculate angular velocity===
  omega = 0.5 * 2 * ((diff_orn).as_quat())[:3]
  if diff_orn.as_quat()[3] < 0:
    omega = -omega
  local_omega = robot_orn.apply(omega) * 10
  local_omega_norm = np.linalg.norm(local_omega)
  if local_omega_norm > 1:
    local_omega /= np.linalg.norm(local_omega)
  # ===calculate action===
  action = np.zeros(5)
  action[0] = -(local_omega[2] - local_omega[1] * np.tan(pitch))
  action[1] = -local_omega[0]
  action[3] = -local_omega[0]
  action[2] = local_omega[1] / np.cos(pitch)
  action[4] = local_omega[1] / np.cos(pitch)
  # compensate for the drop down
  roller_orn_local = R.from_euler('x', pitch)
  roller_orn = roller_orn_local * robot_orn.inv()
  obj_x, obj_y, obj_z = obs['prop/position']
  obj_z -= 0.05
  obj_pos_local = roller_orn.apply([obj_x, obj_y, obj_z])
  action[2] += obj_pos_local[2] * 1
  action[4] -= obj_pos_local[2] * 1
  return action

def ezpolicy_old(obs):
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
  if pitch_err_l > 0.01:
    action[0] = np.clip(pitch_err_l*PITCH_POS2VEL, 0, 1)
  elif pitch_err_l < -0.01:
    action[0] = np.clip(pitch_err_l*PITCH_POS2VEL, -1, 0)
  else:
    action[0] = 0
  if pitch_err_r > 0.01:
    action[2] = np.clip(pitch_err_r*PITCH_POS2VEL, 0, 1)
  elif pitch_err_r < -0.01:
    action[2] = np.clip(pitch_err_r*PITCH_POS2VEL, -1, 0)
  else:
    action[2] = 0
  if abs(pitch_err_l) < 0.01 and abs(pitch_err_r) < 0.01:
    roll_err = target_roll - current_roll
    if roll_err > 0.01:
      action[1] = np.clip(roll_err*ROLL_POS2VEL, 0, 1)
    elif roll_err < -0.01:
      action[1] = np.clip(roll_err*ROLL_POS2VEL, -1, 0)
    action[3] = action[1]
  action[1] += obj_disp * 100
  action[3] -= obj_disp * 100
  return action