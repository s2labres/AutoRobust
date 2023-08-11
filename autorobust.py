import argparse
import gym
import numpy as np
import pandas as pd
import os, sys
import optuna
from envs.PspaceEnv import PspaceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def objective(trial):
    """
    Hyperparam optimization of the Problem Space Rl-based hardening
    """

    # Reward
    rew = trial.suggest_categorical('rew', [1,2,3])
    rew = 1
    # Nubmer of Attack - Retrain iterations
    noa = 15
    # Perturbation budget ~ agent steps
    # steps = trial.suggest_int("p_budget", 50, 200)
    steps = 1000
    # Xplanation Threshold
    xplt = 0.95
    # Learning rate
    lr = trial.suggest_categorical('lr', [0.003,0.001,0.0003,0.0001])
    lr = 0.003

    cwd = '/home/kylesa/avast_clf/v0.2/'
    seed = 2
    arch = 64
    ts = 3e4

    res = []

    # Attack - Retrain iterations
    for i in range(noa):
        print("Iteration: " + str(i+1))
        adv = False if i == 0 else True
        # Scale timesteps
        if i >= 2 and i <= 6: ts = ts*1.5
        # Scale xplainer threshold
        xplt -= 0.01
        # ts = min(ts,9e4)

        # Make training env
        env = gym.make("PspaceEnv-v0",
                    steps=steps,
                    adv=adv,
                    reward=rew,
                    xplt=xplt,)

        # define new agent or load previous one
        # if i == 0:   
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=1024,
            batch_size=32,
            learning_rate=lr,
            gamma=0.99,
            seed=seed,
            policy_kwargs=dict(net_arch=[arch,arch]))
        # else:
        #     model = PPO.load(cwd + 'agents/pspace' + str(i-1) + '.pt', env)

        # maybe load past agent model

        env.reset()
        model.learn(total_timesteps=int(ts), progress_bar=True)
        env.write_hashes()
        sz, rt = env.summary()

        # retrain
        cacc, racc = env.retrain()

        print("avg target score: ", sz[0])
        print("avg mods: ", sz[1])
        print("avg reward: ", sz[2])
        print("resets: ", sz[3])
        # print("actions: ", rt[0])
        print("explanations: ", rt[1])
        print("sizes: ", rt[2][0], rt[2][2])
        sz.extend([rt[2][0], rt[2][2], cacc, racc])
        res.append(sz)
        # write results
        df = pd.DataFrame(res, columns = ['avg_score', 'avg_mods', 'avg_reward', 'resets', 'og size', 'mod size', 'clean acc', 'robust acc'])
        df.to_csv(cwd + "res.csv", index=False, float_format='%.3f')
        model.save(cwd + 'agents/pspace' + str(i) + '.pt')

        # # remove adv reports
        # acwd = cwd + "data_smp/"
        # pta = os.listdir(acwd)
        # for item in pta:
        #     if item.endswith(".json"):
        #         os.remove(os.path.join(acwd, item))

        # garbage collection
        del env
        del model

    return cacc, racc

def test(agt, rew):
    steps = 2000
    cwd = '/home/kylesa/avast_clf/v0.2/'
    seed = 2
    arch = 64
    ts = 3e4

    # Make training env
    env = gym.make("PspaceEnv-v0",
                steps=steps,
                adv=True,
                train=False,
                reward=rew,)

    # Load agent
    model = PPO.load(cwd + 'agents/pspace' + str(agt) + '.pt', env)

    # Evaluate policy
    a, b = evaluate_policy(model, env, n_eval_episodes=10)
    sz, rt = env.summary()
    print("avg target score: ", sz[0])
    print("avg mods + adds: ", sz[1])
    print("avg reward: ", sz[2])
    print("resets: ", sz[3])
    print("actions: ", rt[0])
    print("explanations: ", rt[1])
    print("sizes: ", rt[2])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoRobust Hyperparams")
    parser.add_argument('--rew', default=int(1), type=int, help="Reward used")
    parser.add_argument('--train', default=bool(True), type=bool, help="Train or Test")
    parser.add_argument('--load', default=str("4"), type=str, help="Agent to load")\
    # parser.add_argument()
    args = parser.parse_args()
    if args.train:
        # Create a new optuna study.
        study = optuna.create_study(directions=['maximize', 'maximize'])
        study.optimize(objective, n_trials=1, n_jobs=1, gc_after_trial=True)
    else:
        test(args.load, args.rew)