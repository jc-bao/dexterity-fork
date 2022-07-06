from pathlib import Path
import gym

__version__ = "0.0.15"

# Path to the root of the project.
_PROJECT_ROOT: Path = Path(__file__).parent.parent

# Path to the root of the src files, i.e. `dexterity/`.
_SRC_ROOT: Path = _PROJECT_ROOT / "dexterity"

gym.register(
  id="Roller-v0",
  entry_point="dexterity.utils.dm2gym:GymEnv",
  kwargs={"domain_name": "roller", "task_name": "state_dense"},
)