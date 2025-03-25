import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy

class ZeldaEnv(gym.Env):
    def __init__(self, rom_path):
        super(ZeldaEnv, self).__init__()

        # 初始化 PyBoy
        self.pyboy = PyBoy(rom_path)
        # 定义状态空间
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([255, 255, 9999, 255, 65535, 65535, 255, 255, 255, 255, 255, 255, 255, 255]),
            dtype=np.uint8
        )

        # 定义动作空间
        self.action_space = spaces.Discrete(10)  # 10 个动作

    def reset(self, seed=None):
        # 重置游戏状态
        self.pyboy.reset()
        state = self._get_state()
        return state, {}

    def step(self, action):
        # 执行动作
        self._take_action(action)

        # 获取新的状态
        state = self._get_state()

        # 计算奖励
        reward = self._get_reward()

        # 检查是否结束
        done = self._is_done()

        return state, reward, done, False, {}

    def _get_state(self):
        # 从内存中读取关键变量
        current_health = self.pyboy.get_memory_value(0xDB5A)
        max_health = self.pyboy.get_memory_value(0xDB5B)
        rupees = self.pyboy.get_memory_value(0xDB5D) * 256 + self.pyboy.get_memory_value(0xDB5E)
        position = self.pyboy.get_memory_value(0xDBAE)
        equipment = self.pyboy.get_memory_value(0xDB00) * 256 + self.pyboy.get_memory_value(0xDB01)
        inventory = [self.pyboy.get_memory_value(0xDB02 + i) for i in range(10)]
        keys = self.pyboy.get_memory_value(0xDBD0)
        arrows = self.pyboy.get_memory_value(0xDB45)
        bombs = self.pyboy.get_memory_value(0xDB4D)
        magic_powder = self.pyboy.get_memory_value(0xDB4C)
        secret_shells = self.pyboy.get_memory_value(0xDB0F)
        golden_leaves = self.pyboy.get_memory_value(0xDB15)
        instruments = [self.pyboy.get_memory_value(0xDB65 + i) for i in range(8)]
        map_status = [self.pyboy.get_memory_value(0xD800 + i) for i in range(256)]

        # 组合成状态向量
        state = np.array(
            [current_health, max_health, rupees, position, equipment] + inventory + [keys, arrows, bombs, magic_powder,
                                                                                     secret_shells,
                                                                                     golden_leaves] + instruments + map_status,
            dtype=np.uint8)
        return state

    def _take_action(self, action):
        # 将动作映射到按钮操作
        if action == 1:
            self.pyboy.send_input("A")
        elif action == 2:
            self.pyboy.send_input("B")
        # 其他动作...
        self.pyboy.tick()  # 更新游戏状态

    def _get_reward(self):
        # 根据游戏状态计算奖励
        # 例如：击败敌人 +10，失去生命 -10
        return 0

    def _is_done(self):
        # 检查游戏是否结束
        return False

    def close(self):
        self.pyboy.stop()

# 使用环境
env = ZeldaEnv(
    rom_path='/home/leslie/da_chuang/Legend_of_Zelda/Legend_of_Zelda.gb')
state, _ = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # 随机选择动作
    state, reward, done, _, _ = env.step(action)

    if done:
        break

env.close()