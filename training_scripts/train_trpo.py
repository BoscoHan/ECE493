import os
import gym
import slimevolleygym

from mpi4py import MPI
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import bench, logger, TRPO
from stable_baselines.common.callbacks import EvalCallback

NUM_TIMESTEPS = int(19000000)
SEED = 831
EVAL_FREQ = 200000
EVAL_EPISODES = 1000
LOGDIR = "trpo_mpi"

def make_env(seed):
  env = gym.make("SlimeVolley-v0")
  env.seed(seed)
  return env

def train():
  """
  Train TRPO model for slime volleyball, in MPI multiprocessing. 
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

  model = TRPO(MlpPolicy, env, gamma=0.99, timesteps_per_batch=1024, vf_stepsize=0.0003, vf_iters=3, lam=0.95, verbose=1)

  model.learn(total_timesteps=NUM_TIMESTEPS)

  env.close()
  del env
  if rank == 0:
    model.save("trpo_slime_volleyball")


if __name__ == '__main__':
  train()