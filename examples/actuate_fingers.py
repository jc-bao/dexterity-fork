from typing import List

import imageio
import numpy as np
from dm_control import mjcf

from shadow_hand import shadow_hand_e
from shadow_hand import shadow_hand_e_constants as consts


def render(
    physics: mjcf.Physics,
    frames: List[np.ndarray] = [],
    duration: float = 2.0,
    framerate: float = 30,
) -> None:
    print(duration)
    while physics.data.time < duration:
        physics.step()
        if len(frames) < physics.data.time * framerate:
            pixels = physics.render(width=640, height=480, camera_id="cam0")
            frames.append(pixels)


def main() -> None:
    hand = shadow_hand_e.ShadowHandSeriesE(actuation=consts.Actuation.POSITION)
    physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)

    duration_per = 0.2
    framerate = 30
    frames: List[np.ndarray] = []

    # Create a sequence of control commands that will lower each finger sequentially,
    # then raise each finger in reverse order.
    control = np.zeros((consts.NUM_JOINTS,), dtype=float)
    time = 0.0

    # Wave wrist.
    time += duration_per
    for _ in range(2):
        control[0] = consts.ACTUATOR_CTRLRANGE[consts.Actuators.A_WRJ1][0]
        hand.set_position_control(physics, hand.joint_positions_to_control(control))
        render(physics, frames, time, framerate)
        control[0] = consts.ACTUATOR_CTRLRANGE[consts.Actuators.A_WRJ1][1]
        hand.set_position_control(physics, hand.joint_positions_to_control(control))
        render(physics, frames, time + duration_per, framerate)
        time += 2 * duration_per

    control[0] = 0
    hand.set_position_control(physics, hand.joint_positions_to_control(control))
    render(physics, frames, time, framerate)

    # Thumb.
    time += duration_per
    control[23] = consts.ACTUATOR_CTRLRANGE[consts.Actuators.A_THJ0][1]
    hand.set_position_control(physics, hand.joint_positions_to_control(control))
    render(physics, frames, time, framerate)

    # First finger.
    time += duration_per
    control[4] += np.radians(90)
    control[5] += np.radians(90)
    hand.set_position_control(physics, hand.joint_positions_to_control(control))
    render(physics, frames, time, framerate)

    # Middle finger.
    time += duration_per
    control[8] += np.radians(90)
    control[9] += np.radians(90)
    hand.set_position_control(physics, hand.joint_positions_to_control(control))
    render(physics, frames, time, framerate)

    # Ring finger.
    time += duration_per
    control[12] += np.radians(90)
    control[13] += np.radians(90)
    hand.set_position_control(physics, hand.joint_positions_to_control(control))
    render(physics, frames, time, framerate)

    # Little finger.
    time += duration_per
    control[17] += np.radians(90)
    control[18] += np.radians(90)
    hand.set_position_control(physics, hand.joint_positions_to_control(control))
    render(physics, frames, time, framerate)

    # Little finger.
    time += duration_per
    control[17] -= np.radians(90)
    control[18] -= np.radians(90)
    hand.set_position_control(physics, hand.joint_positions_to_control(control))
    render(physics, frames, time, framerate)

    # Ring finger.
    time += duration_per
    control[12] -= np.radians(90)
    control[13] -= np.radians(90)
    hand.set_position_control(physics, hand.joint_positions_to_control(control))
    render(physics, frames, time, framerate)

    # Middle finger.
    time += duration_per
    control[8] -= np.radians(90)
    control[9] -= np.radians(90)
    hand.set_position_control(physics, hand.joint_positions_to_control(control))
    render(physics, frames, time, framerate)

    # First finger.
    time += duration_per
    control[4] -= np.radians(90)
    control[5] -= np.radians(90)
    hand.set_position_control(physics, hand.joint_positions_to_control(control))
    render(physics, frames, time, framerate)

    # Thumb.
    time += duration_per
    control[23] = consts.ACTUATOR_CTRLRANGE[consts.Actuators.A_THJ0][0]
    hand.set_position_control(physics, hand.joint_positions_to_control(control))
    render(physics, frames, time, framerate)

    imageio.mimsave("temp/render.gif", frames, fps=framerate)


if __name__ == "__main__":
    main()
