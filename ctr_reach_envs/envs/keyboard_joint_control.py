from pynput.keyboard import Key, Listener
import gym
import numpy as np
import ctr_reach_envs
import time


class KeyboardControl(object):
    def __init__(self, env):
        self.env = env

        self.key_listener = Listener(on_press=self.on_press_callback)
        self.key_listener.start()

        self.action = np.zeros_like(self.env.action_space.low)
        self.extension_actions = np.zeros(3)
        self.rotation_actions = np.zeros(3)
        self.reset_prompt = False

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
            if key.char == 'r':
                self.reset_prompt = True
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
            if self.reset_prompt:
                obs = self.env.reset()
                self.reset_prompt = False
            for _ in range(10):
                self.action[:3] = self.extension_actions
                self.action[3:] = self.rotation_actions
                # print('action: ', self.action)
                observation, reward, done, truncated, info = self.env.step(self.action)
                self.extension_actions = np.zeros(3)
                self.rotation_actions = np.zeros(3)
                self.action = np.zeros_like(self.env.action_space.low)
                self.env.render()

        self.env.close()


if __name__ == '__main__':
    env = gym.make('CTR-Reach-v0')
    keyboard_agent = KeyboardControl(env)
    keyboard_agent.run()
