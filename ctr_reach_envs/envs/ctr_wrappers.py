import gym
import numpy as np

from collections import OrderedDict
from ctr_utils.obs_utils import normalize
from ctr_utils.ctr_system_sampling import sample_tube_parameters
import importlib
from ctr_system_parameters import two_tube_ctr_systems, three_tube_ctr_systems


class DiscreteParametersWrapper(gym.Wrapper):
    def __init__(self, env, ctr_systems):
        self.env = env
        self.num_systems = len(ctr_systems.keys())
        # Create a list of the systems
        self.ctr_systems = [ctr_systems[x] for x in ctr_systems]
        self.curr_system = 0
        self.env.ctr_parameters = self.ctr_systems[self.curr_system]
        # Get low and high parameters for observation space
        max_params = []
        min_params = []
        tube_parameters = ['length', 'length_curved', 'diameter_inner', 'diameter_outer', 'stiffness',
                           'torsional_stiffness', 'x_curvature']
        for params in tube_parameters:
            for tube in self.ctr_systems[0].keys():
                max_params.append(max([self.ctr_systems[i][tube][params] for i in range(self.num_systems)]))
                min_params.append(min([self.ctr_systems[i][tube][params] for i in range(self.num_systems)]))

        self.tube_parameters_low = np.array(min_params)
        self.tube_parameters_high = np.array(max_params)
        if isinstance(self.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(low=np.concatenate((env.observation_space.low,
                                                                        -1.0 * np.ones(len(self.tube_parameters_low)))),
                                                    high=np.concatenate((env.observation_space.high,
                                                                         1.0 * np.ones(
                                                                             len(self.tube_parameters_high)))),
                                                    dtype=np.float64)
        elif isinstance(self.observation_space.spaces, OrderedDict):
            self.observation_space['observation'] = gym.spaces.Box(
                low=np.concatenate((env.observation_space['observation'].low,
                                    -1.0 * np.ones(len(
                                        self.tube_parameters_low)))),
                high=np.concatenate((env.observation_space['observation'].high,
                                     1.0 * np.ones(len(
                                         self.tube_parameters_high)))),
                dtype=np.float64)

    def reset(self, **kwargs):
        """Reset function from env but include the tube parameters in reset."""
        self.curr_system = np.random.randint(self.num_systems)
        self.env.set_ctr_system(self.ctr_systems[self.curr_system])
        obs, info = self.env.reset()
        tube_params = self.get_parameters(self.env.get_ctr_system())
        # Concatenate to obs depending on if OrderedDict or regular array
        if isinstance(self.observation_space, gym.spaces.Box):
            param_obs = obs
            param_obs = np.concatenate((param_obs, tube_params))
        elif isinstance(self.observation_space.spaces, OrderedDict):
            param_obs = obs
            param_obs['observation'] = np.concatenate((param_obs['observation'], tube_params))
        return param_obs, info

    def step(self, action):
        next_obs, reward, terminal, truncated, info = self.env.step(action)
        # Get tube parameters
        tube_params = self.get_parameters(self.env.get_ctr_system())
        # Concatenate to obs depending on if OrderedDict or regular array
        if isinstance(self.observation_space, gym.spaces.Box):
            param_obs = next_obs
            param_obs = np.concatenate((param_obs, tube_params))
        elif isinstance(self.observation_space.spaces, OrderedDict):
            param_obs = next_obs
            param_obs['observation'] = np.concatenate((param_obs['observation'], tube_params))
        return param_obs, reward, terminal, truncated, info

    def get_parameters(self, ctr_parameters):
        tube_parameters = ['length', 'length_curved', 'diameter_inner', 'diameter_outer', 'stiffness',
                           'torsional_stiffness', 'x_curvature']
        curr_params = []
        for params in tube_parameters:
            for tube in ctr_parameters.keys():
                curr_params.append(ctr_parameters[tube][params])
        # Normalize tube parameters and return
        return normalize(self.tube_parameters_low, self.tube_parameters_high, np.array(curr_params))


class ContinuousParametersWrapper(gym.Wrapper):
    def __init__(self, env, tube_parameters_low, tube_parameters_high, num_discrete, num_tubes):
        """Observation includes tube parameters. Can be used for generalization of tubes."""
        super(ContinuousParametersWrapper, self).__init__(env)
        assert num_tubes in [2,3]
        self.num_tubes = num_tubes
        self.env = env
        self.tube_parameters_low_dict = tube_parameters_low
        self.tube_parameters_high_dict = tube_parameters_high
        max_params = []
        min_params = []
        tube_parameters = ['length', 'length_curved', 'diameter_inner', 'diameter_outer', 'stiffness',
                           'torsional_stiffness', 'x_curvature']

        for params in tube_parameters:
            if self.num_tubes == 2:
                tube_dict = ['inner', 'outer']
            else:
                tube_dict = ['inner', 'middle', 'outer']
            for tube in tube_dict:
                min_params.append(self.tube_parameters_low_dict[params])
                max_params.append(self.tube_parameters_high_dict[params])

        self.tube_parameters_low = np.array(min_params)
        self.tube_parameters_high = np.array(max_params)
        # TODO: How to deal with achieved and desired goal being so large
        if isinstance(self.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(low=np.concatenate((env.observation_space.low,
                                                                        -1.0 * np.ones(len(self.tube_parameters_low)))),
                                                    high=np.concatenate((env.observation_space.high,
                                                                         1.0 * np.ones(
                                                                             len(self.tube_parameters_high)))),
                                                    dtype=np.float64)
        elif isinstance(self.observation_space.spaces, OrderedDict):
            self.observation_space['observation'] = gym.spaces.Box(
                low=np.concatenate((env.observation_space['observation'].low,
                                    -1.0 * np.ones(len(
                                        self.tube_parameters_low)))),
                high=np.concatenate((env.observation_space['observation'].high,
                                     1.0 * np.ones(len(
                                         self.tube_parameters_high)))),
                dtype=np.float64)

    def reset(self, **kwargs):
        """Reset function from env but include the tube parameters in reset."""
        # Sample new ctr_parameters
        ctr_system = sample_tube_parameters(self.tube_parameters_low_dict, self.tube_parameters_high_dict, num_discrete, num_tubes)
        self.env.set_ctr_system(ctr_system)
        tube_params = self.get_parameters(self.env.get_ctr_system())
        # Set parameters in original environment
        obs, info = self.env.reset()
        # Concatenate to obs depending on if OrderedDict or regular array
        if isinstance(self.observation_space, gym.spaces.Box):
            param_obs = obs
            param_obs = np.concatenate((param_obs, tube_params))
        elif isinstance(self.observation_space.spaces, OrderedDict):
            param_obs = obs
            param_obs['observation'] = np.concatenate((param_obs['observation'], tube_params))
        return param_obs, info

    def step(self, action):
        next_obs, reward, terminal, truncated, info = self.env.step(action)
        # Get tube parameters
        tube_params = self.get_parameters(self.env.get_ctr_system())
        # Concatenate to obs depending on if OrderedDict or regular array
        if isinstance(self.observation_space, gym.spaces.Box):
            param_obs = next_obs
            param_obs = np.concatenate((param_obs, tube_params))
        elif isinstance(self.observation_space.spaces, OrderedDict):
            param_obs = next_obs
            param_obs['observation'] = np.concatenate((param_obs['observation'], tube_params))
        return param_obs, reward, terminal, truncated, info

    def get_parameters(self, ctr_parameters):
        tube_parameters = ['length', 'length_curved', 'diameter_inner', 'diameter_outer', 'stiffness',
                           'torsional_stiffness', 'x_curvature']
        curr_params = []
        for params in tube_parameters:
            for tube in ctr_parameters.keys():
                curr_params.append(ctr_parameters[tube][params])
        # Normalize tube parameters and return
        normalized_curr_params = normalize(self.tube_parameters_low, self.tube_parameters_high, np.array(curr_params))
        assert np.all(normalized_curr_params <= 1.0)
        assert np.all(normalized_curr_params >= -1.0)
        return normalized_curr_params



if __name__ == '__main__':
    # Testing wrappers
    import ctr_reach_envs
    from stable_baselines3.common.env_checker import check_env

    spec = gym.spec('CTR-Reach-v0')
    kwargs = dict()
    tube_parameters_low = {'length': 10e-3, 'length_curved': 10.0e-3, 'diameter_inner': 0.1e-3,
                           'diameter_outer': 0.1e-3, 'stiffness': 5.0e+9,
                           'torsional_stiffness': 1.0e+10, 'x_curvature': 1.0}
    # Sampling of ctr system diameters is a bit tricky.
    # We set the innermost diameter as a sample from the tube_parameters_low['diameter_inner'] to tube_parameters_low['diameter_inner'] + 0.5e-3
    # Then each subsequent tube has a tube seperation value for between tubes and a diameter difference from diameter inner to diameter outer
    # This leads to the maximum diameter of outer being diameter_inner + 0.5e-3 + 1.4e-3
    tube_parameters_high = {'length': 500e-3, 'length_curved': 500.0e-3, 'diameter_inner': 0.1e-3 + 0.5e-3 + 1.4e-3,
                            'diameter_outer': 0.1e-3 + 0.5e-3 + 1.4e-3, 'stiffness': 50.0e+10,
                            'torsional_stiffness': 30.0e+10, 'x_curvature': 25.0}
    num_discrete = 50
    num_tubes = 3
    if num_tubes == 2:
        ctr_system = two_tube_ctr_systems
    else:
        ctr_system = three_tube_ctr_systems
    env = ContinuousParametersWrapper(spec.make(**kwargs), tube_parameters_low, tube_parameters_high, num_discrete, num_tubes)
    #env = DiscreteParametersWrapper(spec.make(**kwargs), ctr_system)

    check_env(env, warn=True)

    for _ in range(10):
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)
            env.render()
