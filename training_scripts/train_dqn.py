# debug
from pdb import set_trace as bp

import os
import gym
import slimevolleygym
from slimevolleygym import SurvivalRewardEnv
# from gym.wrappers import MyWrapper

from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback


# slime volley ball default action space was this: self.action_space = spaces.MultiBinary(3)
# referenced: https://github.com/pfnet/pfrl/blob/master/examples/slimevolley/train_rainbow.pyclass MultiBinaryAsDiscreteAction(gym.ActionWrapper):
# class MultiBinaryAsDiscreteAction(gym.ActionWrapper):
#     """Transforms MultiBinary action space to Discrete.
#     If the action space of a given env is `gym.spaces.MultiBinary(n)`, then
#     the action space of the wrapped env will be `gym.spaces.Discrete(2^n)`,
#     this covers all the combinations of the original action space.
#     Args:
#         env (gym.Env): action space is `gym.spaces.MultiBinary`.
#     """
#     def __init__(self, env):
#         super().__init__(env)
#         assert isinstance(env.action_space, gym.spaces.MultiBinary)
#         self.orig_action_space = env.action_space
#         self.action_space = gym.spaces.Discrete(2 ** env.action_space.n)

#     def action(self, action):
#         arr = [(action >> i) % 2 for i in range(self.orig_action_space.n)]
#         return arr


NUM_TIMESTEPS = int(300000)
SEED = 721
EVAL_FREQ = 1000
EVAL_EPISODES = 1000
LOGDIR = "dqn" # moved to zoo afterwards.

logger.configure(folder=LOGDIR)

env = gym.make("SlimeVolleyNoFrameskip-v0")
# env = MultiBinaryAsDiscreteAction(env)
env.seed(SEED)
    

# # use discreate action space
# # self.action_space = spaces.Box(0, 1.0, shape=(3,))
model = DQN(CnnPolicy, env, gamma=0.99, learning_rate=0.0008, exploration_fraction=0.1, train_freq=4, batch_size=32, double_q=True, target_network_update_freq=1000, verbose=1)

model.learn(total_timesteps=NUM_TIMESTEPS)

# model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.
model.save("dqn_slime_volleyball")
env.close()


