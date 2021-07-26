import os
import gym
import slimevolleygym

from mpi4py import MPI
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import bench, logger, A2C
from stable_baselines.common.callbacks import EvalCallback

NUM_TIMESTEPS = int(150000)
SEED = 831
EVAL_FREQ = 200000
EVAL_EPISODES = 1000
LOGDIR = "a2c_mpi"

def make_env(seed):
  env = gym.make("SlimeVolley-v0")
  env.seed(seed)
  return env

def train():
  """
  Train A2C model for slime volleyball, in MPI multiprocessing. 
  """
  rank = MPI.COMM_WORLD.Get_rank()

  if rank == 0:
    logger.configure(folder=LOGDIR)

  else:
    logger.configure(format_strs=[])
  workerseed = SEED + 10000 * MPI.COMM_WORLD.Get_rank()
  set_global_seeds(workerseed)
  env = make_env(workerseed)

  env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
  env.seed(workerseed)

  model = A2C(MlpPolicy, env, gamma=0.99, vf_coef=0.25, learning_rate=0.005, verbose=1, alpha=0.99)
  model.learn(total_timesteps=NUM_TIMESTEPS)

  env.close()
  del env
  if rank == 0:
    model.save("a2c_slime_volleyball")


if __name__ == '__main__':
  train()

