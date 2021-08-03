from maze.maze_env import Maze
# from RL_brainsample_hacky_PI import rlalgorithm as rlalg1

import numpy as np
import sys
import matplotlib.pyplot as plt
from stable_baselines.common.env_checker import check_env
import pickle
import time
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv

import warnings
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import os
import numpy as np
import argparse
from time import sleep

# debug
from pdb import set_trace as bp
np.set_printoptions(threshold=20, precision=4, suppress=True, linewidth=200)

# class PPOPolicy:
#   def __init__(self, path):
#     self.model = PPO1.load(path)
#   def predict(self, obs):
#     action, state = self.model.predict(obs, deterministic=True)
#     return action

# class A2C:
#   def __init__(self, path):
#     self.model = A2C.load(path)
#   def predict(self, obs):
#     action, state = self.model.predict(obs, deterministic=True)
#     return action

class TRPO:
  def __init__(self, path):
    self.model = TRPO.load(path)
  def predict(self, obs):
    action, state = self.model.predict(obs, deterministic=True)
    return action

# class DQN:
#   def __init__(self, path):
#     self.model = DQN.load(path)
#   def predict(self, obs):
#     action, state = self.model.predict(obs, deterministic=True)

#     action_arr = np.array([0, 0, 0])
#     action_arr[action % env.action_space.n] = action
#     return action_arr

# class RandomPolicy:
#   def __init__(self, path):
#     self.action_space = gym.spaces.MultiBinary(3)
#     pass
#   def predict(self, obs):
#     return self.action_space.sample()


if __name__ == "__main__":
    sim_speed = 0.05

    #Example Short Fast for Debugging
    showRender=True
    episodes=40
    renderEveryNth=5
    printEveryNth=1
    do_plot_rewards=True

    #Example Full Run, you may need to run longer
    # showRender=False
    # episodes=2000
    # renderEveryNth=10000
    # printEveryNth=100
    # do_plot_rewards=True

    if(len(sys.argv)>1):
        episodes = int(sys.argv[1])
    if(len(sys.argv)>2):
        showRender = sys.argv[2] in ['true','True','T','t']
    if(len(sys.argv)>3):
        datafile = sys.argv[3]

    # Task Specifications
    agentXY=[0,0]

    # Task 2 - C
    wall_shape=np.array([[2,2],[3, 2], [4, 2], 
                         [2,3], [2, 4], [2,6], [2, 7],
                         [3, 8], [4, 8], [5, 8]])
    pits=np.array([[5,2],[2,8]])
    goalXY=[2,5]


    # env1 = Maze(agentXY,goalXY,wall_shape, pits)
    # RL1 = rlalg1(actions=list(range(env1.n_actions)))
    # data1={}
    # env1.after(10, update(env1, RL1, data1, episodes))
    # env1.mainloop()
    # experiments = [(env1,RL1, data1)]

    experiments = []

    env = gym.make('MazeWorld-v0', agentXY = agentXY, goalXY = goalXY, walls = wall_shape, pits = pits)
    # RL2 = rlalg2(actions=list(range(env2.n_actions)))
    data2={}

    model = DQN(MlpPolicy, env, gamma=0.99, learning_rate=0.0008, exploration_fraction=0.1, train_freq=4, batch_size=32, double_q=True, target_network_update_freq=1000, verbose=1)

    model.learn(total_timesteps=15000)
    model.save("dqn_MazeWorld-v0")


    
    # env2.after(10, update(env2, RL2, data2, episodes))
    # env2.mainloop()
    # # experiments.append((env2,RL2, data2))
    # experiments = [(env2,RL2, data2)]

    print("All experiments complete")

    for env, RL, data in experiments:
        print("{} : max reward = {} medLast100={} varLast100={}".format(RL.display_name, np.max(data['global_reward']),np.median(data['global_reward'][-100:]), np.var(data['global_reward'][-100:])))




# TODO: put the learning stuff in the main func above:
def train_TRPO(self, env):
    agentXY=[0,0]

    # Task 2 - C
    wall_shape=np.array([[2,2],[3, 2], [4, 2], 
                            [2,3], [2, 4], [2,6], [2, 7],
                            [3, 8], [4, 8], [5, 8]])
    pits=np.array([[5,2],[2,8]])
    goalXY=[2,5]

    env = gym.make('MazeWorld-v0', agentXY = agentXY, goalXY = goalXY, walls = wall_shape, pits = pits)

    model = TRPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("trpo_MazeWorld-v0")

    del model # remove to demonstrate saving and loading

    model = TRPO.load("trpo_MazeWorld-v0")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()





def rollout(env, policy0, render_mode=False):

  obs0 = env.reset()

  done = False
  total_reward = 0
  #count = 0

  while not done:

    action0 = policy0.predict(obs0)
    # bp()

    obs0, reward, done, info = env.step(action0)
    total_reward += reward

    if render_mode:
      env.render()
      """ # used to render stuff to a gif later.
      img = env.render("rgb_array")
      filename = os.path.join("gif","daytime",str(count).zfill(8)+".png")
      cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
      count += 1
      """
      sleep(0.01)
    return total_reward
