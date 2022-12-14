import gym
from ctr_utils.obs_utils import *
from ctr_utils.render_utils import Rendering
from ctr_utils.goal_tolerance import GoalTolerance
from ctr_kinematics.ThreeTubeCTR import ThreeTubeCTRKinematics


class CtrReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, ctr_parameters, goal_parameters, reward_type, joint_representation, noise_parameters,
                 initial_joints, max_extension_action, max_rotation_action, steps_per_episode, n_substeps, render_mode):
        # TODO: How to include multiple tube systems in observation for training
        # Tubes ordered outermost to innermost in environment but innermost to outermost in kinematics
        self.ctr_parameters = ctr_parameters
        self.tube_length = np.array(
            [ctr_parameters['outer']['length'], ctr_parameters['middle']['length'],
             ctr_parameters['inner']['length']])
        self.num_tubes = len(self.ctr_parameters)
        assert self.num_tubes in [2, 3]
        self.goal_parameters = goal_parameters
        self.reward_type = reward_type
        assert joint_representation in ['proprioceptive', 'egocentric']
        self.joint_representation = joint_representation
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
        self.render_mode = render_mode
        self.visualization = None
        # Assume delta goal min and maxes for now

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def reset(self, seed=None, options=None):
        # TimeLimit wrapper to ensure end of episode is handled correctly
        self.desired_goal = self.kinematics.forward_kinematics(flip_joints(sample_joints(self.tube_length)))
        self.achieved_goal = self.kinematics.forward_kinematics(flip_joints(self.joints))
        obs = get_obs(self.joints, self.joint_representation, self.desired_goal, self.achieved_goal,
                      self.goal_tolerance.current_tol, self.tolerance_min_max, self.tube_length)
        return obs, {'achieved_goal': self.achieved_goal, 'desired_goal': self.desired_goal}

    def step(self, action):
        assert not np.all(np.isnan(action))
        assert self.action_space.contains(action)
        for i in range(self.n_substeps):
            self.joints = apply_action(action, self.max_extension_action, self.max_rotation_action, self.joints,
                                       self.tube_length)
        achieved_goal = self.kinematics.forward_kinematics(flip_joints(self.joints))
        reward = self.compute_reward(achieved_goal, self.desired_goal, {'reward_type': self.reward_type})
        done = np.linalg.norm(achieved_goal - self.desired_goal, axis=-1) < self.goal_tolerance.current_tol
        obs = get_obs(self.joints, self.joint_representation,
                      self.desired_goal, achieved_goal, self.goal_tolerance.current_tol, self.tolerance_min_max,
                      self.tube_length)
        info = {'achieved_goal': achieved_goal, 'desired_goal': self.desired_goal}
        return obs, reward, done, False, info

    @staticmethod
    def compute_distance(achieved_goal, desired_goal):
        assert achieved_goal.shape == desired_goal.shape
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return d

    def compute_reward(self, achieved_goal, desired_goal, info):
        dist = self.compute_distance(achieved_goal, desired_goal)
        if self.reward_type == 'dense':
            reward = -1.0 - 1.0 * dist
        else:
            reward = -1.0 * (dist > self.goal_tolerance.current_tol)
        return reward

    def render(self):
        if self.render_mode == 'human':
            if self.visualization is None:
                self.visualization = Rendering()
            self.visualization.render(self.achieved_goal, self.desired_goal, self.kinematics.r1, self.kinematics.r2,
                                      self.kinematics.r3)

    def close(self):
        if self.visualization is not None:
            self.visualization.close()
            self.visualization = None


if __name__ == '__main__':
    import ctr_reach_envs

    spec = gym.spec('CTR-Reach-v0')
    kwargs = dict()
    env = spec.make(**kwargs)

    env.reset()
    for _ in range(150):
        action = env.action_space.sample()
        env.step(action)
        env.render()
