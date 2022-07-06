import skvideo.io
import numpy as np

from dexterity.utils.dm2gym import GymEnv
from dexterity import manipulation


if __name__=='__main__':
  env = GymEnv(domain_name="roller", task_name="state_dense")
  obs = env.reset()
  imgs = []
  for _ in range(100):
    act = manipulation.ezpolicy(obs)
    # act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
    render_im = env.render(mode='rgb_array')
    imgs.append(render_im)
  skvideo.io.vwrite('vid/gym.mp4', np.array(imgs))