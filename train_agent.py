import argparse
import gym
import numpy as np
import os, sys
import optuna
from envs.FspaceEnv import FspaceEnv
from stable_baselines3 import PPO
import cProfile
import pstats

steps = 100
seed = 2
arch = 32
ts = 3e4

cwd = '/home/kylesa/avast_clf/v0.2/'

# Make training env
env = gym.make("FspaceEnv-v0",
               steps=steps,)
    
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=32,
    learning_rate=0.0003,
    gamma=0.99,
    seed=seed,
    policy_kwargs=dict(net_arch=dict(vf=[arch,arch], pi=[arch,arch]))
)

def main():
    env.reset()
    # for i in range(100):
    #     env.step(2)
    # env.printrep()
    model.learn(total_timesteps=int(ts), progress_bar=True)
    # env.write_hashes()
    # model.save(cwd + 'agents/fspace1.pt')

main()

# p = cProfile.run('main()', 'profile_stats')
# stats = pstats.Stats('profile_stats')
# stats.sort_stats('tottime')
# stats.print_stats(10)