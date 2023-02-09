from gym.envs.registration import register
import numpy as np

# New hardware parameters, check stiffness and diameters
ctr_parameters = {
    'inner':
        {'length': 3.0 * 96.41e-3 + 153.95e-3 + 90.3e-3, 'length_curved': 90e-3, 'diameter_inner': 0.508e-3,
         'diameter_outer': 0.66e-3, 'stiffness': 7.5e+10, 'torsional_stiffness': 2.5e+10,
         'x_curvature': 1 / 40.63 * 1e3, 'y_curvature': 0},

    'middle':
        {'length': 82.19e-3 + 87.5e-3, 'length_curved': 87.5e-3, 'diameter_inner': 0.7e-3,
         'diameter_outer': 1.0e-3, 'stiffness': 7.5e+10, 'torsional_stiffness': 2.5e+10,
         'x_curvature': 1 / 52.3 * 1e3, 'y_curvature': 0},

    'outer':
        {'length': 11.72e-3 + 61.03e-3, 'length_curved': 61.03e-3, 'diameter_inner': 1.15e-3,
         'diameter_outer': 1.63e-3, 'stiffness': 7.5e+12, 'torsional_stiffness': 2.5e+12,
         'x_curvature': 1 / 71.23 * 1e3, 'y_curvature': 0}
}
register(
    id='CTR-Reach-v0', entry_point='ctr_reach_envs.envs:CtrReachEnv',
    kwargs={
        'ctr_parameters': ctr_parameters,
        'goal_parameters': {
            'final_tol': 0.001, 'initial_tol': 0.020, 'function_steps': 500000, 'function_type': 'constant'
        },
        'noise_parameters': {},
        'reward_type': 'dense',
        'joint_representation': 'proprioceptive',
        'initial_joints': np.array([0., 0., 0., 0., 0., 0.]),
        'resample_joints': True,
        'home_offset': np.array([50.75e-3, 119.69e-3, 235.96e-3 + 2.0 * 96.41e-3]),
        'max_retraction': np.array([22.0e-3, 50.0e-3, 97.0e-3]),
        'max_rotation': np.pi / 3,
        'max_extension_action': 0.001,
        'max_rotation_action': np.deg2rad(5.0),
        'steps_per_episode': 200,
        'n_substeps': 10,
        'render_mode': 'human'
    },
    max_episode_steps=200
)

register(
    id='CTR-Reach-Goal-v0', entry_point='ctr_reach_envs.envs:CtrReachGoalEnv',
    kwargs={
        'ctr_parameters': ctr_parameters,
        'goal_parameters': {
            'final_tol': 0.001, 'initial_tol': 0.020, 'function_steps': 500000, 'function_type': 'constant'
        },
        'noise_parameters': {},
        'reward_type': 'sparse',
        'joint_representation': 'proprioceptive',
        'initial_joints': np.array([0., 0., 0., 0., 0., 0.]),
        'home_offset': np.array([50.75e-3, 119.69e-3, 235.96e-3 + 2.0 * 96.41e-3]),
        'max_retraction': np.array([22.0e-3, 50.0e-3, 97.0e-3]),
        'max_rotation': np.pi / 3,
        'resample_joints': True,
        'max_extension_action': 0.001,
        'max_rotation_action': np.deg2rad(5.0),
        'steps_per_episode': 200,
        'n_substeps': 10,
        'render_mode': 'human'
    },
    max_episode_steps=200
)
