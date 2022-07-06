import gym
from gym import spaces
from dm_env import specs
import io
import cv2
import uuid
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
from dm_robotics.transformations import transformations as tr

from dexterity import manipulation


class OpenCVImageViewer():
  """A simple OpenCV highgui based dm_control image viewer
  This class is meant to be a drop-in replacement for
  `gym.envs.classic_control.rendering.SimpleImageViewer`
  """

  def __init__(self, *, escape_to_exit=False):
    """Construct the viewing window"""
    self._escape_to_exit = escape_to_exit
    self._window_name = str(uuid.uuid4())
    cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
    self._isopen = True

  def __del__(self):
    """Close the window"""
    cv2.destroyWindow(self._window_name)
    self._isopen = False

  def imshow(self, img):
    """Show an image"""
    # Convert image to BGR format
    cv2.imshow(self._window_name, img[:, :, [2, 1, 0]])
    # Listen for escape key, then exit if pressed
    if cv2.waitKey(1) in [27] and self._escape_to_exit:
      exit()

  @property
  def isopen(self):
    """Is the window open?"""
    return self._isopen

  def close(self):
    pass


def convert_dm_control_to_gym_space(dm_control_space):
  r"""Convert dm_control space to gym space. """
  if isinstance(dm_control_space, specs.BoundedArray):
    space = spaces.Box(low=dm_control_space.minimum,
                       high=dm_control_space.maximum,
                       dtype=dm_control_space.dtype)
    assert space.shape == dm_control_space.shape
    return space
  elif isinstance(dm_control_space, specs.Array) and not isinstance(dm_control_space, specs.BoundedArray):
    space = spaces.Box(low=-float('inf'),
                       high=float('inf'),
                       shape=dm_control_space.shape,
                       dtype=dm_control_space.dtype)
    return space
  elif isinstance(dm_control_space, dict):
    space = spaces.Dict({key: convert_dm_control_to_gym_space(value)
                         for key, value in dm_control_space.items()})
    return space


class GymEnv(gym.Env):
  def __init__(self, domain_name, task_name):
    self.env = manipulation.load(domain_name, task_name)
    self.metadata = {'render.modes': ['human', 'rgb_array'],
                     'video.frames_per_second': round(1.0 / self.env.control_timestep())}

    self.observation_space = convert_dm_control_to_gym_space(
      self.env.observation_spec())
    self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())
    self.viewer = None
    self.current_obs = {'goal_state': np.zeros(6)}
    self.orn_errors_list = np.zeros(70)
    self.step_cnt = 0

  def seed(self, seed):
    return self.env.task.random.seed(seed)

  def step(self, action):
    timestep = self.env.step(action)
    observation = timestep.observation
    reward = timestep.reward
    done = timestep.last()
    info = {}
    if done: self.step_cnt = 0
    if self.current_obs['goal_state'][0] != observation['goal_state'][0]: 
      self.step_cnt = 0
      self.orn_errors_list = np.zeros(70)
    else: 
      self.step_cnt += 1
    self.current_obs = observation
    return observation, reward, done, info

  def reset(self):
    timestep = self.env.reset()
    self.current_obs = timestep.observation
    return timestep.observation

  def render(self, mode='human', **kwargs):
    if 'camera_id' not in kwargs:
      kwargs['camera_id'] = 0  # Tracking camera

    goal_orn = R.from_quat(np.concatenate([self.current_obs['goal_state'][...,1:], self.current_obs['goal_state'][...,[0]]], axis=-1))
    obj_orn = R.from_quat(
      np.concatenate([self.current_obs['prop/orientation'][...,1:], self.current_obs['prop/orientation'][...,[0]]], axis=-1))
    orn_error = goal_orn * obj_orn.inv()
    err_quat = orn_error.as_quat()
    err_quat = np.concatenate([err_quat[...,1:], err_quat[...,[0]]], axis=-1)
    self.orn_errors_list[self.step_cnt] = abs(tr.quat_angle(err_quat)-np.pi)
    plt.plot(self.orn_errors_list)
    plt.axvline(x = self.step_cnt, color = 'r')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=20)
    plt.clf()
    plt_im = np.array(Image.open(img_buf))[..., :3]
    img = self.env.physics.render(**kwargs)
    img[:plt_im.shape[0], :plt_im.shape[1], :] = plt_im
    if mode == 'rgb_array':
      return img
    elif mode == 'human':
      if self.viewer is None:
        self.viewer = OpenCVImageViewer()
      self.viewer.imshow(img)
      return self.viewer.isopen
    else:
      raise NotImplementedError

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None
    return self.env.close()