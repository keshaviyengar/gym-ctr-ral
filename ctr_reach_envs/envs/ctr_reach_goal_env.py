#import gymnasium as gym
import gym
from ctr_reach_envs.envs.ctr_reach_env import CtrReachEnv
from ctr_utils.obs_utils import *


# This gym environment creates a GoalEnv gym environment based on the existing reach env. Specifically, it creates the
# required dictionary style observation.


class CtrReachGoalEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    def __init__(self, ctr_parameters, goal_parameters, reward_type, joint_representation, noise_parameters,
                 initial_joints, resample_joints, max_extension_action, max_rotation_action, home_offset, max_retraction,
                 max_rotation, steps_per_episode, n_substeps, render_mode=None):
        self.env = CtrReachEnv(ctr_parameters, goal_parameters, reward_type, joint_representation, noise_parameters,
                               initial_joints, resample_joints, max_extension_action, max_rotation_action, home_offset,
                               max_retraction, max_rotation, steps_per_episode, n_substeps,
                               render_mode)
        self.observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(low=-np.ones(3), high=np.ones(3), dtype=float),
            achieved_goal=gym.spaces.Box(low=-np.ones(3), high=np.ones(3), dtype=float),
            observation=self.env.observation_space
        ))
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed, options)
        obs_dict = convert_obs_to_dict(obs, info['achieved_goal'], info['desired_goal'])
        return obs_dict, {}

    def seed(self, seed=None):
        self.env.seed(seed)

    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(action)
        obs_dict = convert_obs_to_dict(obs, info['achieved_goal'], info['desired_goal'])
        reward = self.compute_reward(obs_dict['achieved_goal'], obs_dict['desired_goal'], {'reward_type': self.env.reward_type})
        return obs_dict, reward, terminal, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        dist = self.env.compute_distance(achieved_goal, desired_goal)
        if self.env.reward_type == 'dense':
            reward = -1.0 - 1.0 * dist
        else:
            reward = -1.0 * (dist > self.env.goal_tolerance.current_tol)
        return reward

    def set_ctr_system(self, ctr_system):
        self.env.set_ctr_system(ctr_system)

    def get_ctr_system(self):
        return self.env.get_ctr_system()

    def update_goal_tolerance(self, num_timesteps):
        self.env.update_goal_tolerance(num_timesteps)

    def get_goal_tolerance(self):
        return self.env.get_goal_tolerance()

    def render(self):
        self.env.render()


if __name__ == '__main__':
    import ctr_reach_envs

    spec = gym.spec('CTR-Reach-Goal-v0')
    kwargs = dict()
    env = spec.make(**kwargs)

    env.reset()
    for _ in range(150):
        action = env.action_space.sample()
        env.step(action)
        env.render()
