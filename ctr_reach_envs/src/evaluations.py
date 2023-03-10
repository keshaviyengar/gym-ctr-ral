import gym
import numpy as np
import time
import csv
from stable_baselines3 import PPO
import ctr_reach_envs
from ctr_reach_envs.src.utils import run_episode
#from ctr_reach_envs.envs.ctr_wrappers import ROSWrapper

def drl_ik_solver(env, model, episode, desired_goal=None, initial_joints=None, output_path=None):
    # Move to starting position
    time.sleep(0.5)
    achieved_goals, desired_goals, qs, r1, r2, r3 = run_episode(env, model, desired_goal)
    #with open(output_path, 'w+') as f:
    #    writer = csv.writer(f)
    #    step = 0
    #    for ag, dg, joint in zip(achieved_goals, desired_goals, qs):
    #        error = np.linalg.norm(ag - dg)
    #        step += 1
    #        writer.writerow(data_line(0, step, joint, ag, dg, np.array([error])))

if __name__ == '__main__':
    # Load in agent
    exp_name = 'ppo_full_noise_5_deg_20'
    model_path = '/home/keshav/ctm2-stable-baselines/ral-2023/gym-ctr-reach-ral/ctr_reach_envs/src/saved_policies/' \
                 + exp_name + '/rl_model_100000_steps'
    env_kwargs = {
        'goal_parameters': {
            'final_tol': 0.001, 'initial_tol': 0.001, 'function_steps': 500000, 'function_type': 'constant'
        }
    }
    model = PPO.load(model_path)
    env = gym.make('CTR-Reach-v0', **env_kwargs)
    num_episodes = 1
    for episode in range(num_episodes):
        drl_ik_solver(env, model, episode)
