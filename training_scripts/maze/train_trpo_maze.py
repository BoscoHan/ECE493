import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO
import sys


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

# del model 

model = TRPO.load("trpo_MazeWorld-v0")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()



# didn't work
