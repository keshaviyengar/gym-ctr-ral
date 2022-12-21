import gym
import numpy as np
from ctr_utils.obs_utils import *
from ctr_utils.render_utils import Rendering
from ctr_utils.goal_tolerance import GoalTolerance
from ctr_kinematics.ctr_kinematics.ThreeTubeCTR import ThreeTubeCTRKinematics


class CtrReachEnv(gym.Env):
    def __init__(self, ctr_parameters, goal_parameters, noise_parameters, initial_joints, max_extension_action,
                 max_rotation_action, steps_per_episode, n_substeps):
        # TODO: How to include multiple tube systems in observation for training
        # Tubes ordered outermost to innermost in environment but innermost to outermost in kinematics
        self.ctr_parameters = ctr_parameters
        self.tube_length = np.array(
            [ctr_parameters['tube_0']['length'], ctr_parameters['tube_1']['length'], ctr_parameters['tube_2']['length']])
        self.num_tubes = len(self.ctr_parameters)
        assert self.num_tubes in [2, 3]
        self.goal_parameters = goal_parameters
        self.noise_parameters = noise_parameters
        self.joints = initial_joints
        self.max_extension_action = max_extension_action
        self.max_rotation_action = max_rotation_action
        self.steps_per_episode = steps_per_episode
        self.n_substeps = n_substeps
        self.action_space = gym.spaces.Box(low=-1.0 * np.ones(self.num_tubes * 2),
                                           high=1.0 * np.ones(self.num_tubes * 2), dtype=np.float)
        # Set observation space (num_tubes * trig_representation + del_xyz + tolerance)
        self.observation_space = gym.spaces.Box(low=-1.0 * np.ones(self.num_tubes * 3 + 3 + 1),
                                                high=1.0 * np.ones(self.num_tubes * 3 + 3 + 1), dtype=np.float)

        self.achieved_goal = None
        self.desired_goal = None
        self.goal_tolerance = GoalTolerance(goal_parameters['initial_tol'], goal_parameters['final_tol'],
                                            goal_parameters['function_steps'], goal_parameters['function_type'])
        self.tolerance_min_max = np.array([self.goal_tolerance.final_tol, self.goal_tolerance.init_tol])

        self.kinematics = ThreeTubeCTRKinematics(ctr_parameters)
        self.visualization = None
        # Assume delta goal min and maxes for now

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def reset(self):
        # TimeLimit wrapper to ensure end of episode is handled correctly
        self.desired_goal = self.kinematics.forward_kinematics(flip_joints(sample_joints(self.tube_length)))
        self.joints = sample_joints(self.tube_length)
        self.achieved_goal = self.kinematics.forward_kinematics(flip_joints(self.joints))
        return get_obs(self.joints, self.desired_goal, self.achieved_goal, self.goal_tolerance.current_tol,
                       self.tolerance_min_max, self.tube_length)

    def step(self, action):
        assert not np.all(np.isnan(action))
        assert self.action_space.contains(action)
        for i in range(self.n_substeps):
            self.joints = apply_action(action, self.max_extension_action, self.max_rotation_action, self.joints,
                                       self.tube_length)
        achieved_goal = self.kinematics.forward_kinematics(flip_joints(action))
        reward = self.compute_reward(achieved_goal, self.desired_goal, self.goal_tolerance.current_tol)
        done = np.linalg.norm(achieved_goal - self.desired_goal, axis=-1) < self.goal_tolerance.current_tol
        obs = get_obs(self.joints, self.desired_goal, achieved_goal, self.goal_tolerance.current_tol, self.tolerance_min_max,
                      self.tube_length)
        info = {}
        return obs, reward, done, info

    @staticmethod
    def compute_reward(achieved_goal, desired_goal, goal_tolerance):
        assert achieved_goal.shape == desired_goal.shape
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -float((d > goal_tolerance))

    def render(self, mode='human'):
        if mode == 'live':
            if self.visualization is None:
                self.visualization = Rendering()
            self.visualization.render(self.achieved_goal, self.desired_goal, self.kinematics.r1, self.kinematics.r2,
                                      self.kinematics.r3)

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None


if __name__ == '__main__':
    ctr_parameters = {
        'tube_0':
            {'length': 110.0e-3, 'length_curved': 100e-3, 'diameter_inner': 1.2e-3, 'diameter_outer': 1.5e-3,
             'stiffness': 50e+10, 'torsional_stiffness': 50.0e+10 / (2 * (1 + 0.3)),
             'x_curvature': 4.37, 'y_curvature': 0},
        'tube_1':
            {'length': 165.0e-3, 'length_curved': 100e-3, 'diameter_inner': 0.7e-3, 'diameter_outer': 0.9e-3,
             'stiffness': 50e+10, 'torsional_stiffness': 50.0e+10 / (2 * (1 + 0.3)),
             'x_curvature': 12.4, 'y_curvature': 0},
        'tube_2':
            {'length': 210.0e-3, 'length_curved': 31e-3, 'diameter_inner': 0.4e-3, 'diameter_outer': 0.5e-3,
             'stiffness': 50e+10, 'torsional_stiffness': 50.0e+10 / (2 * (1 + 0.3)),
             'x_curvature': 28.0, 'y_curvature': 0},
    }

    goal_tolerance_parameters = {
        'final_tol': 0.001, 'initial_tol': 0.020, 'function_steps': 200000, 'function_type': 'constant'
    }

    # TODO: Complete noise parameters
    noise_parameters = {}

    initial_joints = np.array([0., 0., 0., 0., 0., 0.])
    max_extension_action = 0.001
    max_rotation_action = np.deg2rad(5.0)
    steps_per_episode = 150
    n_substeps = 10

    env = CtrReachEnv(ctr_parameters, goal_tolerance_parameters, noise_parameters, initial_joints, max_extension_action,
                      max_rotation_action, steps_per_episode, n_substeps)
    env.reset()
    action = env.action_space.sample()
    env.step(action)
    env.render('live')
