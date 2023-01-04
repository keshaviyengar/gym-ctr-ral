from pynput.keyboard import Key, Listener
import gym
import numpy as np
from ctr_reach_env import CtrReachEnv


class KeyboardControl(object):
    def __init__(self, env):
        self.env = env

        self.key_listener = Listener(on_press=self.on_press_callback)
        self.key_listener.start()

        self.action = np.zeros_like(self.env.action_space.low)
        self.extension_actions = np.zeros(3)
        self.rotation_actions = np.zeros(3)

        self.extension_value = self.env.action_space.high[0] / 2
        self.rotation_value = self.env.action_space.high[-1] / 2
        self.exit = False

    def on_press_callback(self, key):
        # Tube 1 (inner most tube) is w s a d
        # Tube 2 (outer most tube) is t g f h
        # Tube 3 (outer most tube) is i k j l
        try:
            if key.char in ['w', 's', 'a', 'd']:
                if key.char == 'w':
                    self.extension_actions[0] = self.extension_value
                elif key.char == 's':
                    self.extension_actions[0] = -self.extension_value
                elif key.char == 'a':
                    self.rotation_actions[0] = self.rotation_value
                elif key.char == 'd':
                    self.rotation_actions[0] = -self.rotation_value
            if key.char in ['t', 'g', 'f', 'h']:
                if key.char == 't':
                    self.extension_actions[1] = self.extension_value
                elif key.char == 'g':
                    self.extension_actions[1] = -self.extension_value
                elif key.char == 'f':
                    self.rotation_actions[1] = self.rotation_value
                elif key.char == 'h':
                    self.rotation_actions[1] = -self.rotation_value
            if key.char in ['i', 'k', 'j', 'l']:
                if key.char == 'i':
                    self.extension_actions[2] = self.extension_value
                elif key.char == 'k':
                    self.extension_actions[2] = -self.extension_value
                elif key.char == 'j':
                    self.rotation_actions[2] = self.rotation_value
                elif key.char == 'l':
                    self.rotation_actions[2] = -self.rotation_value
        except AttributeError:
            if key == Key.esc:
                self.exit = True
                exit()
            else:
                self.extension_actions = np.zeros(3)
                self.rotation_actions = np.zeros(3)

    def run(self):
        obs = self.env.reset()
        while not self.exit:
            self.action[:3] = self.extension_actions
            self.action[3:] = self.rotation_actions
            # print('action: ', self.action)
            observation, reward, done, info = self.env.step(self.action)
            self.extension_actions = np.zeros(3)
            self.rotation_actions = np.zeros(3)
            self.action = np.zeros_like(self.env.action_space.low)
            self.env.render('live')
        self.env.close()


if __name__ == '__main__':
    ctr_parameters = {
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
             'x_curvature': 4.37, 'y_curvature': 0},
    }

    goal_tolerance_parameters = {
        'final_tol': 0.001, 'initial_tol': 0.020, 'function_steps': 200000, 'function_type': 'constant'
    }

    # TODO: Complete noise parameters
    noise_parameters = {}
    reward_type = 'dense'
    joint_representation = 'proprioceptive'

    initial_joints = np.array([0., 0., 0., 0., 0., 0.])
    max_extension_action = 0.001
    max_rotation_action = np.deg2rad(5.0)
    steps_per_episode = 150
    n_substeps = 10

    env = CtrReachEnv(ctr_parameters, goal_tolerance_parameters, reward_type, joint_representation,
                      noise_parameters, initial_joints, max_extension_action, max_rotation_action, steps_per_episode,
                      n_substeps)
    keyboard_agent = KeyboardControl(env)
    keyboard_agent.run()
