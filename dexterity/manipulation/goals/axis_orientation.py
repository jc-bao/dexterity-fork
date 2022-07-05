from this import s
import numpy as np
from dm_control import mjcf
from dm_control.composer.variation import rotations
from dm_control.entities.props import primitive
from dm_env import specs
from dm_robotics.transformations import transformations as tr

from dexterity import goal


class QuaternionFromAxisAngle(rotations.base.Variation):
  """Quaternion variation specified in terms of variations in axis and angle."""
  def __call__(self, random_state=None):
    random_state = random_state or np.random
    # self.alpha = random_state.uniform(0, 2 * np.pi)
    # self.beta = random_state.uniform(0, np.pi/2)
    # axis = np.array([np.cos(self.beta)*np.cos(self.alpha), np.cos(self.beta)*np.sin(self.alpha), np.sin(self.beta)])
    self.theta = random_state.uniform(0, np.pi)
    axis = np.array([0, np.cos(self.theta), np.sin(self.theta)])
    # axis = np.array([1,0,0])
    axis = axis / np.linalg.norm(axis)
    angle = random_state.uniform(-np.pi, np.pi)
    # angle = np.pi/2
    sine, cosine = np.sin(angle), np.cos(angle)
    # record the axis and angle for debug purposes
    self.angle = angle
    self.axis = axis
    return np.array([cosine, axis[0] * sine, axis[1] * sine, axis[2] * sine])


class AxisOrientation(goal.GoalGenerator):
  def __init__(
      self,
      prop: primitive.Primitive,
      name: str = "axis_orientation_goal_generator",
  ) -> None:
    super().__init__()

    self._name = name
    self._prop = prop
    self._sampler = QuaternionFromAxisAngle()

  def goal_spec(self) -> specs.Array:
    return specs.Array(shape=(4,), dtype=np.float64, name=self._name)

  def initialize_episode(
      self, physics: mjcf.Physics, random_state: np.random.RandomState
  ) -> None:
    del physics, random_state  # Unused.

  def current_state(self, physics: mjcf.Physics) -> np.ndarray:
    return np.array(physics.bind(self._prop.orientation).sensordata)

  def next_goal(
      self, physics: mjcf.Physics, random_state: np.random.RandomState
  ) -> np.ndarray:
    del physics  # Unused.
    return self._sampler(random_state)

  def relative_goal(
      self, goal_state: np.ndarray, current_state: np.ndarray
  ) -> np.ndarray:
    return tr.quat_diff_active(source_quat=current_state, target_quat=goal_state)

  def goal_distance(
      self, goal_state: np.ndarray, current_state: np.ndarray
  ) -> np.ndarray:
    err_quat = self.relative_goal(goal_state, current_state)
    err_axisangle = tr.quat_to_axisangle(err_quat)
    return np.linalg.norm(err_axisangle, keepdims=True)  # type: ignore

  @property
  def name(self) -> str:
    return self._name
