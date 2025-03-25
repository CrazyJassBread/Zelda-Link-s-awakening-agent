import gym
from gym import spaces
import numpy as np
from pyboy import PyBoy
import uuid


class ZeldaEnv(gym.Env):
    def __init__(self, rom_path, config=None):
        super(ZeldaEnv, self).__init__()

        # 初始化 PyBoy
        self.pyboy = PyBoy(rom_path)

        # 配置初始化
        self.s_path = config.get("session_path", "./")  # 存储路径
        self.save_video = config.get("save_video", False)  # 是否保存视频
        self.headless = config.get("headless", False)  # 是否无界面模式
        self.frame_stacks = 3  # 堆叠的帧数
        self.action_freq = config.get("action_freq", 4)  # 动作频率
        self.max_steps = config.get("max_steps", 10000)  # 最大步骤数
        self.reset_count = 0  # 重置计数器

        # 定义动作空间：8个动作（4个方向 + 4个按钮）
        self.action_space = spaces.Discrete(8)

        # 定义观察空间
        self.observation_space = spaces.Dict(
            {
                "health": spaces.Box(low=0, high=14, shape=(1,), dtype=np.uint8),  # 当前生命值（0-14颗心）
                "max_health": spaces.Box(low=0, high=14, shape=(1,), dtype=np.uint8),  # 最大生命值（最大14颗心）
                "rupees": spaces.Box(low=0, high=999, shape=(1,), dtype=np.uint16),  # 当前卢比数量
                "position": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),  # 当前坐标（位置）
                "inventory": spaces.MultiBinary(10),  # 库存物品（最多 10 个物品）
                "keys": spaces.Box(low=0, high=99, shape=(1,), dtype=np.uint8),  # 钥匙数量
                "arrows": spaces.Box(low=0, high=99, shape=(1,), dtype=np.uint8),  # 弓箭数量
                "bombs": spaces.Box(low=0, high=99, shape=(1,), dtype=np.uint8),  # 炸弹数量
                "magic_powder": spaces.Box(low=0, high=99, shape=(1,), dtype=np.uint8),  # 魔法粉数量
                "map_progress": spaces.MultiBinary(256),  # 已访问的地图区域（每个区域为一个标志位）
                "secrets": spaces.Box(low=0, high=999, shape=(1,), dtype=np.uint16),  # 秘密贝壳数量
                "golden_leaves": spaces.Box(low=0, high=99, shape=(1,), dtype=np.uint8),  # 金叶数量
                "instruments": spaces.MultiBinary(8),  # 已获得的乐器（8个地牢）
            }
        )

    def reset(self):
        # 重置游戏状态
        self.pyboy.reset()  # 重新启动游戏
        state = self._get_state()  # 获取当前游戏状态
        self.reset_count += 1  # 增加重置计数器
        return state, {}

    def step(self, action):
        # 执行动作
        self._take_action(action)

        # 获取新的游戏状态
        state = self._get_state()

        # 计算奖励
        reward = self._get_reward()

        # 判断游戏是否结束
        done = self._is_done()

        return state, reward, done, {}

    def _get_state(self):
        # 从内存中读取游戏状态
        current_health = self.pyboy.memory[0xDB5A]  # 当前生命值
        max_health = self.pyboy.memory[0xDB5B]  # 最大生命值
        rupees = self.pyboy.memory[0xDB5D] * 256 + self.pyboy.memory[0xDB5E]  # 当前卢比数量
        position = self.pyboy.memory[0xDBAE]  # 当前位置（地牢中的位置）
        inventory = [self.pyboy.memory[0xDB02 + i] for i in range(10)]  # 库存物品（最多 10 个物品）
        keys = self.pyboy.memory[0xDBD0]  # 钥匙数量
        arrows = self.pyboy.memory[0xDB45]  # 弓箭数量
        bombs = self.pyboy.memory[0xDB4D]  # 炸弹数量
        magic_powder = self.pyboy.memory[0xDB4C]  # 魔法粉数量
        secrets = self.pyboy.memory[0xDB0F]  # 秘密贝壳数量
        golden_leaves = self.pyboy.memory[0xDB15]  # 金叶数量
        instruments = [self.pyboy.memory[0xDB65 + i] for i in range(8)]  # 已获得的乐器（8个地牢）
        map_progress = [self.pyboy.memory[0xD800 + i] for i in range(256)]  # 已访问的地图区域

        # 将所有信息汇总为状态字典
        state = {
            "health": np.array([current_health]),
            "max_health": np.array([max_health]),
            "rupees": np.array([rupees]),
            "position": np.array([position]),
            "inventory": np.array(inventory),
            "keys": np.array([keys]),
            "arrows": np.array([arrows]),
            "bombs": np.array([bombs]),
            "magic_powder": np.array([magic_powder]),
            "map_progress": np.array(map_progress),
            "secrets": np.array([secrets]),
            "golden_leaves": np.array([golden_leaves]),
            "instruments": np.array(instruments)
        }
        return state

    def _take_action(self, action):
        # 根据动作执行相应的操作
        valid_actions = [
            "down", "left", "right", "up", "a", "b", "start", "select"
        ]
        self.pyboy.send_input(valid_actions[action])

        # 更新游戏状态
        self.pyboy.tick(self.action_freq)

    def _get_reward(self):
        # 计算奖励（例如健康值、卢比数量等）
        current_health = self.pyboy.memory[0xDB5A]
        max_health = self.pyboy.memory[0xDB5B]
        rupees = self.pyboy.memory[0xDB5D] * 256 + self.pyboy.memory[0xDB5E]

        # 假设奖励与健康值和卢比数量成正比
        reward = current_health + (rupees / 10)
        return reward

    def _is_done(self):
        # 判断游戏是否结束
        # 假设游戏结束条件是健康值为 0
        current_health = self.pyboy.memory[0xDB5A]
        return current_health == 0

    def close(self):
        # 关闭 PyBoy 实例
        self.pyboy.stop()


config = {
    "session_path": "D:/codes/codes_pycharm/da_chuang/sessions",  # 存储路径
    "save_video": True,  # 是否保存视频
    "headless": False,  # 是否无界面模式
    "action_freq": 4,  # 动作频率
    "max_steps": 10000,  # 最大步骤数
}

# 初始化环境
env = ZeldaEnv(rom_path="D:/codes/codes_pycharm/da_chuang/Legend_of_Zelda/Legend_of_Zelda.gb", config=config)

# 重置环境并获取初始状态
state, _ = env.reset()

# 模拟智能体的行动
for _ in range(1000):
    action = env.action_space.sample()  # 随机选择一个动作
    state, reward, done, _ = env.step(action)

    # 打印奖励
    print(f"Reward: {reward}")

    if done:
        break

# 关闭环境
env.close()
