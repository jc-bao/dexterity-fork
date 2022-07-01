"""A standalone app for visualizing manipulation tasks."""

from typing import Sequence

from absl import app
from absl import flags
from dm_control import viewer
import numpy as np

from dexterity import manipulation
from dexterity.manipulation.wrappers import ActionNoise

flags.DEFINE_enum(
  "environment_name",
  None,
  manipulation.ALL_NAMES,
  "Optional 'domain_name.task_name' pair specifying the environment to load.",
)
flags.DEFINE_integer("seed", None, "RNG seed.")
flags.DEFINE_bool("timeout", True, "Whether episodes should have a time limit.")
flags.DEFINE_float("action_noise", 0.0, "Action noise scale.")

FLAGS = flags.FLAGS


def prompt_environment_name(values: Sequence[str]) -> str:
  environment_name = None
  while not environment_name:
    environment_name = input("Please enter the environment name: ")
    if not environment_name or environment_name not in values:
      print(f"'{environment_name}' is not a valid environment name.")
      environment_name = None
  return environment_name


def main(_) -> None:
  if FLAGS.environment_name is None:
    print("\n ".join(["Available environments:"] + manipulation.ALL_NAMES))
    environment_name = prompt_environment_name(manipulation.ALL_NAMES)
  else:
    environment_name = FLAGS.environment_name

  index = manipulation.ALL_NAMES.index(environment_name)
  domain_name, task_name = manipulation.ALL_TASKS[index]

  env = manipulation.load(
    domain_name=domain_name,
    task_name=task_name,
    seed=FLAGS.seed,
    time_limit=None if FLAGS.timeout else float("inf"))

  if FLAGS.action_noise > 0.0:
    # By default, the viewer will apply the midpoint action since the action spec is
    # bounded. So this wrapper will add noise centered around this midpoint.
    env = ActionNoise(env, scale=FLAGS.action_noise)

  # Print entity and task observables.
  timestep = env.reset()
  for k, v in timestep.observation.items():
    print(f"{k}: {v.shape}")

  action_spec = env.action_spec()
  def random_policy(timestep):
    del timestep  # Unused
    action = np.random.uniform(action_spec.minimum, action_spec.maximum)
    return action.astype(action_spec.dtype)

  def change_pitch_policy(timestep):
    # current state
    obs = timestep.observation
    joint_pos = obs['roller_hand/joint_positions']
    pitch_r, pitch_l = joint_pos[1], joint_pos[4]
    roll_r, roll_l = joint_pos[2], joint_pos[5]
    # target state
    yz = env.task.goal_generator._sampler.axis[1:]
    target_pitch = np.arctan(yz[0]/yz[1])
    # action
    action = np.zeros(action_spec.shape)
    delta_pitch_r =  target_pitch - pitch_r
    delta_pitch_l =  target_pitch - pitch_l
    action[0] = np.clip(delta_pitch_l, -0.5, 0.5)
    action[2] = np.clip(delta_pitch_r, -0.5, 0.5)
    return action.astype(action_spec.dtype)

  def change_roll_policy(timestep):
    # current state
    obs = timestep.observation
    joint_pos = obs['roller_hand/joint_positions']
    # target state
    target_roll = env.task.goal_generator._sampler.angle
    current_roll = np.arccos(obs['prop/orientation'][0])
    # action
    action = np.zeros(action_spec.shape)
    if target_roll > (current_roll+0.001):
      action[1] = 1
    elif target_roll < (current_roll-0.001):
      action[1] = -1
    action[3] = action[1]
    return action.astype(action_spec.dtype)
  
  def handcrafted_policy(timestep):
    # current state
    obs = timestep.observation
    joint_pos = obs['roller_hand/joint_positions']
    pitch_r, pitch_l = joint_pos[1], joint_pos[4]
    obj_quat = obs['prop/orientation']
    current_roll = np.arccos(obj_quat[0])
    if obj_quat[3] < 0:
      current_roll = -current_roll
    obj_y, obj_z = obs['prop/position'][1:]
    obj_z -= 0.05
    # target state
    yz = env.task.goal_generator._sampler.axis[1:]
    target_pitch = np.arctan(yz[0]/yz[1])
    target_roll = env.task.goal_generator._sampler.angle
    obj_disp = -obj_y * np.cos(target_pitch) + obj_z * np.sin(target_pitch)
    # action
    action = np.zeros(action_spec.shape)
    gamma = 0.5
    pitch_err_l = target_pitch - pitch_r
    pitch_err_r = target_pitch - pitch_l
    if (abs(pitch_err_l) > 0.01) and (abs(pitch_err_r) > 0.01):
      action[0] = target_pitch
      action[2] = target_pitch
    else:
      action[0] = target_pitch
      action[2] = target_pitch
      if target_roll > (current_roll+0.01):
        action[1] = 1
      elif target_roll < (current_roll-0.01):
        action[1] = -1
      action[3] = action[1]
      action[1] += obj_disp*100
      action[3] -= obj_disp*100
    return action.astype(action_spec.dtype)

  import skvideo.io
  for i in range(10):
    print('trail: ', i)
    timestep = env.reset()
    imgs = []
    while not timestep.last():
      action = handcrafted_policy(timestep)
      timestep = env.step(action)
      imgs.append(env.physics.render(camera_id="front_close"))
    if len(imgs) > 0:
      print('write vid: ', i)
      skvideo.io.vwrite(f"vid/{i}.mp4", np.array(imgs))
  # viewer.launch(env, policy = handcrafted_policy)


if __name__ == "__main__":
  app.run(main)
