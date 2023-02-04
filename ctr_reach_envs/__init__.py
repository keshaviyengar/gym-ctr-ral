from gym.envs.registration import register
import numpy as np

register(
    id='CTR-Reach-v0', entry_point='ctr_reach_envs.envs:CtrReachEnv',
    kwargs={
        'ctr_parameters': {
            # 'inner':
            #    {'length': 400e-3, 'length_curved': 200e-3, 'diameter_inner': 2 * 0.35e-3,
            #     'diameter_outer': 2 * 0.55e-3, 'stiffness': 70e+9, 'torsional_stiffness': 10.0e+9, 'x_curvature': 12.0,
            #     'y_curvature': 0},
            # 'outer':
            #    {'length': 300e-3, 'length_curved': 150e-3, 'diameter_inner': 2 * 0.7e-3,
            #     'diameter_outer': 2 * 0.9e-3, 'stiffness': 70e+9, 'torsional_stiffness': 10.0e+9, 'x_curvature': 6.0,
            #     'y_curvature': 0},
            # New hardware parameters, check stiffness and diameters
            # New hardware parameters, check stiffness and diameters
            'inner':
                {'length': 104.7e-3, 'length_curved': 104.7e-3, 'diameter_inner': 0.508e-3,
                 'diameter_outer': 0.66e-3, 'stiffness': 7.5e+10, 'torsional_stiffness': 2.5e+10,
                 'x_curvature': 1 / 40.63 * 1e3, 'y_curvature': 0},

            'middle':
                {'length': 50.0e-3, 'length_curved': 50.0e-3, 'diameter_inner': 0.7e-3,
                 'diameter_outer': 1.0e-3, 'stiffness': 7.5e+10, 'torsional_stiffness': 2.5e+10,
                 'x_curvature': 1 / 52.3 * 1e3, 'y_curvature': 0},

            'outer':
                {'length': 22.0e-3, 'length_curved': 22.0e-3, 'diameter_inner': 1.15e-3,
                 'diameter_outer': 1.63e-3, 'stiffness': 7.5e+12, 'torsional_stiffness': 2.5e+12,
                 'x_curvature': 1 / 71.23 * 1e3, 'y_curvature': 0}
        },
        'goal_parameters': {
            'final_tol': 0.001, 'initial_tol': 0.020, 'function_steps': 500000, 'function_type': 'constant'
        },
        'noise_parameters': {},
        'reward_type': 'sparse',
        'joint_representation': 'proprioceptive',
        'initial_joints': np.array([0., 0., 0., 0., 0., 0.]),
        'resample_joints': True,
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
        'ctr_parameters': {
            'inner':
                {'length': 104.7e-3, 'length_curved': 104.7e-3, 'diameter_inner': 0.508e-3,
                 'diameter_outer': 0.66e-3, 'stiffness': 7.5e+10, 'torsional_stiffness': 2.5e+10,
                 'x_curvature': 1 / 40.63 * 1e3, 'y_curvature': 0},

            'middle':
                {'length': 50.0e-3, 'length_curved': 50.0e-3, 'diameter_inner': 0.7e-3,
                 'diameter_outer': 1.0e-3, 'stiffness': 7.5e+10, 'torsional_stiffness': 2.5e+10,
                 'x_curvature': 1 / 52.3 * 1e3, 'y_curvature': 0},

            'outer':
                {'length': 22.0e-3, 'length_curved': 22.0e-3, 'diameter_inner': 1.15e-3,
                 'diameter_outer': 1.63e-3, 'stiffness': 7.5e+12, 'torsional_stiffness': 2.5e+12,
                 'x_curvature': 1 / 71.23 * 1e3, 'y_curvature': 0}
        },
        'goal_parameters': {
            'final_tol': 0.001, 'initial_tol': 0.020, 'function_steps': 500000, 'function_type': 'constant'
        },
        'noise_parameters': {},
        'reward_type': 'sparse',
        'joint_representation': 'proprioceptive',
        'initial_joints': np.array([0., 0., 0., 0., 0., 0.]),
        'resample_joints': True,
        'max_extension_action': 0.001,
        'max_rotation_action': np.deg2rad(5.0),
        'steps_per_episode': 200,
        'n_substeps': 10,
        'render_mode': 'human'
    },
    max_episode_steps=200
)
