import gym
import ctr_reach_envs
import numpy as np

from stable_baselines3 import DDPG, HER, PPO

def run_episode(env, model, goal=None, max_steps=None):
    if goal is not None:
        obs, info = env.reset(**{'goal': goal})
    else:
        obs, info = env.reset()
    achieved_goals = list()
    desired_goals = list()
    r1 = list()
    r2 = list()
    r3 = list()
    qs = list()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, truncated, infos = env.step(action)
        env.render()
        achieved_goals.append(infos['achieved_goal'])
        desired_goals.append(infos['desired_goal'])
        r1.append(env.kinematics.r1)
        r2.append(env.kinematics.r2)
        r3.append(env.kinematics.r3)
        qs.append(infos['q_achieved'])
        # After each step, store achieved goal as well as rs
        if done or infos.get('is_success', False) or truncated:
            print("Tip Error: " + str(infos.get('error')))
            print("Achieved joint: " + str(infos.get('q_achieved')))
            break
    if infos.get('error') > env.unwrapped.goal_tolerance.current_tol * 1000.0:
        print("Could not get close to starting position...")
    return achieved_goals, desired_goals, qs, r1, r2, r3
