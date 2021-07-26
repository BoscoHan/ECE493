# # debug
# from pdb import set_trace as bp

# import os
# import gym
# import slimevolleygym
# from slimevolleygym import SurvivalRewardEnv
# from gym.wrappers import MyWrapper

# from stable_baselines import DQN
# # from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
# from stable_baselines import logger
# from stable_baselines.common.callbacks import EvalCallback




# NUM_TIMESTEPS = int(25000)
# SEED = 721
# EVAL_FREQ = 250000
# EVAL_EPISODES = 1000
# LOGDIR = "dqn" # moved to zoo afterwards.

# logger.configure(folder=LOGDIR)

# env = gym.make('SlimeVolley-v0')

# env = MyWrapper(env)




# # use discreate action space
# # self.action_space = spaces.Box(0, 1.0, shape=(3,))
# model = DQN(MlpPolicy, env, verbose=2)

# bp()

# # eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

# model.learn(total_timesteps=NUM_TIMESTEPS)

# model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

# env.close()



import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO

env = gym.make('CartPole-v1')

model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("trpo_cartpole")

del model # remove to demonstrate saving and loading

model = TRPO.load("trpo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

