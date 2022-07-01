"""roller hand constants."""

from math import radians as rad
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dexterity import _SRC_ROOT

# Path to the roller hand E series XML file.
ROLLER_HAND_XML: Path = (
  _SRC_ROOT
  / "models"
    / "vendor"
    / "roller_robot"
    / "roller_hand_description"
    / "mjcf"
    / "roller_hand.xml"
)


# ====================== #
# Joint constants
# ====================== #
JOINTS: Tuple[str, ...] = (
  "pitch_l",
  "roll_l",
  "pitch_r",
  "roll_r",
)


# The total number of joints.
NUM_JOINTS: int = len(JOINTS)

JOINT_GROUP: Dict[str, Tuple[str, ...]] = {
  "left": ("pitch_l", "roll_l"),
  "right": ("pitch_r", "roll_r"),
}

# ====================== #
# Actuation constants
# ====================== #

"""Actuators of the roller Hand."""
ACTUATORS: Tuple[str, ...] = (
  "motor_pitch_l",
  "motor_pitch_r",
  "motor_roll_l",
  "motor_roll_r",
)

NUM_ACTUATORS: int = len(ACTUATORS)

ACTUATOR_GROUP: Dict[str, Tuple[str, ...]] = {
  "left": ("motor_pitch_l", "motor_roll_l"),
  "right": ("motor_pitch_r", "motor_roll_r"),
}

ACTUATOR_JOINT_MAPPING: Dict[str, Tuple[str, ...]] = {
  "motor_pitch_l": ("pitch_l",),
  "motor_roll_l": ("roll_l",),
  "motor_pitch_r": ("pitch_r",),
  "motor_roll_r": ("roll_r",),
}

# Reverse mapping of `ACTUATOR_JOINT_MAPPING`.
JOINT_ACTUATOR_MAPPING: Dict[str, str] = {
  v: k for k, vs in ACTUATOR_JOINT_MAPPING.items() for v in vs
}


def _compute_projection_matrices() -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
  position_to_control = np.zeros((NUM_ACTUATORS, NUM_JOINTS))
  control_to_position = np.zeros((NUM_JOINTS, NUM_ACTUATORS))
  coupled_joint_ids = []
  actuator_ids = dict(zip(ACTUATORS, range(NUM_ACTUATORS)))
  joint_ids = dict(zip(JOINTS, range(NUM_JOINTS)))
  for actuator, joints in ACTUATOR_JOINT_MAPPING.items():
    value = 1.0 / len(joints)
    a_id = actuator_ids[actuator]
    j_ids = np.array([joint_ids[joint] for joint in joints])
    if len(joints) > 1:
      coupled_joint_ids.append([joint_ids[joint] for joint in joints])
    position_to_control[a_id, j_ids] = 1.0
    control_to_position[j_ids, a_id] = value
  return position_to_control, control_to_position, coupled_joint_ids


# Projection matrices for mapping control space to joint space and vice versa. These
# matrices should premultiply the vector to be projected.
# POSITION_TO_CONTROL maps a control vector to a joint vector.
# CONTROL_TO_POSITION maps a joint vector to a control vector.
(
  POSITION_TO_CONTROL,
  CONTROL_TO_POSITION,
  COUPLED_JOINT_IDS,
) = _compute_projection_matrices()


"""Tendons of the roller Hand.

These are used to model the underactuation of the *FJ0 and *FJ1 joints of the main
fingers. A tendon is defined for each *FJ0-*FJ1 pair, and an actuator is used to
drive it.
"""
TENDONS: Tuple[str, ...] = (
)
# The total number of tendons.
NUM_TENDONS: int = len(TENDONS)

# ====================== #
# Fingertip constants
# ====================== #

FINGERTIP_NAMES: Tuple[str, ...] = (
  "l3",
  "r3",
)
