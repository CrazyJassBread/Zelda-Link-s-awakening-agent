from sympy import refine
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from pyboy import PyBoy

#actions = ["", "a", "b", "left", "right", "up", "down"]
actions = [""]

DEBUG = False

class ZeldaPyBoyEnv(gym.Env):
    def __init__(self, rom_path, state_path=None, debug=DEBUG):
        self.debug = debug
        super().__init__()
        self.pyboy = PyBoy(rom_path, sound=False)
        # 初始化游戏参数

        if state_path:
            with open(state_path, "rb") as f:
                self.pyboy.load_state(f)

        self.pyboy.set_emulation_speed(1 if not self.debug else 0)

        self.zelda = self.pyboy.game_wrapper
        self.action_space = spaces.Discrete(len(actions))  # 9 种动作
        self.observation_space = spaces.Box(low=0, high=255, shape=(32, 32), dtype=np.uint8)

        self.pyboy.game_wrapper.start_game()
        self.prev_screen = self._get_observation()

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action!"
        if action != 0:
            self.pyboy.button(actions[action])

        self.pyboy.tick(1)
        done = self.pyboy.game_wrapper.game_over
        reward = self._calculate_reward()
        observation = self.pyboy.game_area()
        info = {}

        return observation, reward, done, False, info

    def _calculate_reward(self):
        """计算奖励：基于屏幕变化"""
        new_screen = self._get_observation()
        reward = np.sum(new_screen != self.prev_screen) 
        self.prev_screen = new_screen 
        return reward

    def _get_observation(self):
        """获取游戏区域的画面"""
        return self.pyboy.game_area()  

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pyboy.game_wrapper.reset_game()
        observation = self.pyboy.game_area()
        return observation, {}

    def close(self):
        self.pyboy.stop()
    
    def print_information(self):
        print(self.zelda)

# 运行测试
if __name__ == "__main__":
    import signal
    import os

    def force_exit(sig, frame):
        print("\n强制退出程序...")
        os._exit(0)

    signal.signal(signal.SIGINT, force_exit)
    
    env = ZeldaPyBoyEnv("D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb", "D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda_duyuhan.gb.state")
    obs, info = env.reset()
    
    i = 0
    while True:
        env.pyboy.tick(1)
        if i % 10 == 0:
            action = random.randint(0, len(actions) - 1)
            obs, reward, done, _, _ = env.step(action)
        if i % 50 == 0:
            env.print_information()
        i += 1