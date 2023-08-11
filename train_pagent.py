import argparse
import gym
import numpy as np
import os, sys
import optuna
from envs.PspaceEnv import PspaceEnv
from stable_baselines3 import PPO
import cProfile
import pstats

steps = 100
seed = 2
arch = 64
ts = 3e4

cwd = '/home/kylesa/avast_clf/v0.2/'

# Make training env
env = gym.make("PspaceEnv-v0",
               steps=steps,)
    
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=32,
    learning_rate=0.0003,
    gamma=0.99,
    seed=seed,
    policy_kwargs=dict(net_arch=[arch,arch]),
    # policy_kwargs=dict(net_arch=dict(vf=[arch,arch], pi=[arch,arch]))
)

def main():
    env.reset()
    model.learn(total_timesteps=int(ts), progress_bar=True)
    # env.write_hashes()
    env.summary()
    # model.save(cwd + 'agents/pspace1.pt')

main()

# p = cProfile.run('main()', 'profile_stats')
# stats = pstats.Stats('profile_stats')
# stats.sort_stats('tottime')
# stats.print_stats(10)