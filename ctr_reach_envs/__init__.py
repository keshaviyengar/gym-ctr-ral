from gymnasium.envs.registration import register
import numpy as np

register(
    id='CTR-Reach-v0', entry_point='ctr_reach_envs.envs:CtrReachEnv',
    kwargs={
        'ctr_parameters': {
            'inner':
                {'length': 210.0e-3, 'length_curved': 31e-3, 'diameter_inner': 0.4e-3, 'diameter_outer': 0.5e-3,
                 'stiffness': 50e+10, 'torsional_stiffness': 50.0e+10 / (2 * (1 + 0.3)),
                 'x_curvature': 28.0, 'y_curvature': 0},
            'middle':
                {'length': 165.0e-3, 'length_curved': 100e-3, 'diameter_inner': 0.7e-3, 'diameter_outer': 0.9e-3,
                 'stiffness': 50e+10, 'torsional_stiffness': 50.0e+10 / (2 * (1 + 0.3)),
                 'x_curvature': 12.4, 'y_curvature': 0},
            'outer':
                {'length': 110.0e-3, 'length_curved': 100e-3, 'diameter_inner': 1.2e-3, 'diameter_outer': 1.5e-3,
                 'stiffness': 50e+10, 'torsional_stiffness': 50.0e+10 / (2 * (1 + 0.3)),
                 'x_curvature': 4.37, 'y_curvature': 0}
        },
        'goal_parameters': {
            'final_tol': 0.001, 'initial_tol': 0.020, 'function_steps': 200000, 'function_type': 'constant'
        },
        'noise_parameters': {},
        'reward_type': 'dense',
        'joint_representation': 'proprioceptive',
        'initial_joints': np.array([0., 0., 0., 0., 0., 0.]),
        'max_extension_action': 0.001,
        'max_rotation_action': np.deg2rad(5.0),
        'steps_per_episode': 150,
        'n_substeps': 10,
        'render_mode': 'human'
    },
    max_episode_steps=150
)

register(
    id='CTR-Reach-Goal-v0', entry_point='ctr_reach_envs.envs:CtrReachGoalEnv',
    kwargs={
        'ctr_parameters': {
            'inner':
                {'length': 210.0e-3, 'length_curved': 31e-3, 'diameter_inner': 0.4e-3, 'diameter_outer': 0.5e-3,
                 'stiffness': 50e+10, 'torsional_stiffness': 50.0e+10 / (2 * (1 + 0.3)),
                 'x_curvature': 28.0, 'y_curvature': 0},
            'middle':
                {'length': 165.0e-3, 'length_curved': 100e-3, 'diameter_inner': 0.7e-3, 'diameter_outer': 0.9e-3,
                 'stiffness': 50e+10, 'torsional_stiffness': 50.0e+10 / (2 * (1 + 0.3)),
                 'x_curvature': 12.4, 'y_curvature': 0},
            'outer':
                {'length': 110.0e-3, 'length_curved': 100e-3, 'diameter_inner': 1.2e-3, 'diameter_outer': 1.5e-3,
                 'stiffness': 50e+10, 'torsional_stiffness': 50.0e+10 / (2 * (1 + 0.3)),
                 'x_curvature': 4.37, 'y_curvature': 0}
        },
        'goal_parameters': {
            'final_tol': 0.001, 'initial_tol': 0.020, 'function_steps': 200000, 'function_type': 'constant'
        },
        'noise_parameters': {},
        'reward_type': 'dense',
        'joint_representation': 'proprioceptive',
        'initial_joints': np.array([0., 0., 0., 0., 0., 0.]),
        'max_extension_action': 0.001,
        'max_rotation_action': np.deg2rad(5.0),
        'steps_per_episode': 150,
        'n_substeps': 10,
        'render_mode': 'human'
    },
    max_episode_steps=150
)
