import gym
from gym.envs.registration import register

for env in list(gym.envs.registry.env_specs):
     if 'FspaceEnv-v0' in env:
          del gym.envs.registry.env_specs[env]

for env in list(gym.envs.registry.env_specs):
     if 'PspaceEnv-v0' in env:
          del gym.envs.registry.env_specs[env]

register(
    id='FspaceEnv-v0',
    entry_point='envs.FspaceEnv:FspaceEnv'
   )
register(
    id='PspaceEnv-v0',
    entry_point='envs.PspaceEnv:PspaceEnv'
   )