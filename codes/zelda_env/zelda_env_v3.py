import gym
from gym import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from pathlib import Path

#############################################
# 也能跑，改了奖励函数，加上了钥匙，但是好像没什么用###
#############################################
class ZeldaEnv(gym.Env):
    def __init__(self, rom_path, config=None):
        super(ZeldaEnv, self).__init__()

        # 配置初始化
        self.s_path = Path(config.get("session_path", "./"))  # 存储路径
        self.print_rewards = config.get("print_rewards", False)  # 是否打印奖励
        self.headless = config["headless"]  # 是否无头模式（不显示图形界面）
        self.init_state = config["init_state"]  # 初始状态文件路径
        self.save_video = config["save_video"]  # 是否保存视频
        self.fast_video = config.get("fast_video", False)  # 是否快速视频
        self.frame_stacks = 3  # 堆叠的帧数
        self.action_freq = config.get("action_freq", 4)  # 动作频率
        self.max_steps = config.get("max_steps", 10000000)  # 最大步骤数

        self.s_path.mkdir(exist_ok=True)  # 创建存储路径
        self.full_frame_writer = None  # 完整视频帧写入器
        self.model_frame_writer = None  # 模型视频帧写入器
        self.map_frame_writer = None  # 地图视频帧写入器

        self.reset_count = 0  # 重置计数器
        # 定义动作空间（允许的按钮按下动作）
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        # 定义释放动作
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        # 设置动作空间（动作数量与 valid_actions 一致）
        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.metadata = {"render.modes": []}  # 渲染模式
        self.reward_range = (0, 15000)  # 奖励范围

        # 定义状态空间
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

        # 使用无头模式或SDL2窗口显示
        head = "null" if config["headless"] else "SDL2"

        self.pyboy = PyBoy(  # 初始化 PyBoy 模拟器
            rom_path,  # 游戏 ROM 路径
            window=head,
        )

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)  # 设置模拟器的仿真速度

        # 修改：新增杀敌计数和钥匙获取标志
        self.enemies_killed = 0
        self.key_obtained = False
        self.last_health = 1

    # def reset(self, seed=None, options={}):
    #     self.seed = seed
    #     # 加载游戏的初始状态
    #     try:
    #         with open(self.init_state, "rb") as f:
    #             self.pyboy.load_state(f)  # 加载保存的游戏状态
    #     except FileNotFoundError:
    #         print(f"Warning: Game state file {self.init_state} not found, starting a new game.")
    #
    #     # 初始化地图内存
    #     self.init_map_mem()
    #
    #     # 初始化代理的状态统计
    #     self.agent_stats = []
    #
    #     self.explore_map_dim = (16, 16)  # 假设为 16x16 的地图大小
    #     self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)
    #
    #     self.recent_screens = np.zeros((144, 160, 3), dtype=np.uint8)
    #     self.recent_actions = np.zeros(self.frame_stacks, dtype=np.uint8)
    #
    #     self.levels_satisfied = False
    #     self.base_explore = 0
    #     self.max_opponent_level = 0
    #     self.max_event_rew = 0
    #     self.max_level_rew = 0
    #     self.last_health = 1
    #     self.total_healing_rew = 0
    #     self.died_count = 0
    #     self.party_size = 0
    #     self.step_count = 0
    #
    #     # 初始化事件标志
    #     self.base_event_flags = sum([
    #         self.bit_count(self.read_m(i))
    #         for i in range(0xD747, 0xD87E)  # 假设事件标志的地址范围
    #     ])
    #
    #     self.current_event_flags_set = {}
    #
    #     # 初始化进度奖励
    #     self.max_map_progress = 0
    #     self.progress_reward = self.get_game_state_reward()
    #     self.total_reward = sum([val for _, val in self.progress_reward.items()])
    #     self.reset_count += 1
    #
    #     # 修改：重置杀敌计数和钥匙获取标志
    #     self.enemies_killed = 0
    #     self.key_obtained = False
    #
    #     return self._get_obs(), {}
    def reset(self, seed=None, options={}):
        self.seed = seed
        # 加载游戏的初始状态
        try:
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)  # 加载保存的游戏状态
        except FileNotFoundError:
            print(f"Warning: Game state file {self.init_state} not found, starting a new game.")

        # 初始化地图内存
        self.init_map_mem()

        # 初始化代理的状态统计
        self.agent_stats = []

        self.explore_map_dim = (8, 8)
        self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)

        self.recent_screens = np.zeros((144, 160, 3), dtype=np.uint8)
        self.recent_actions = np.zeros(self.frame_stacks, dtype=np.uint8)

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0

        # 初始化事件标志， 这个是？好像没用到
        self.base_event_flags = sum([
            self.bit_count(self.read_m(i))
            for i in range(0xD747, 0xD87E)  # 假设事件标志的地址范围
        ])

        self.current_event_flags_set = {}

        # 初始化进度奖励
        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()  # 现在是整数
        self.total_reward = self.progress_reward  # 直接赋值
        self.reset_count += 1

        # 重置杀敌计数和钥匙获取标志
        self.enemies_killed = 0
        self.key_obtained = False

        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}

    def _get_obs(self):
        # 获取当前状态
        current_health = self.pyboy.memory[0xDB5A]  # 当前生命值
        max_health = self.pyboy.memory[0xDB5B]  # 最大生命值
        rupees = self.pyboy.memory[0xDB5D] * 256 + self.pyboy.memory[0xDB5E]  # 当前卢比数量
        position = self.pyboy.memory[0xDBAE]  # 当前在地牢中的位置

        # 其他游戏状态项
        inventory = [self.pyboy.memory[0xDB02 + i] for i in range(10)]  # 库存物品
        keys = self.pyboy.memory[0xDBD0]  # 钥匙数量
        arrows = self.pyboy.memory[0xDB45]  # 弓箭数量
        bombs = self.pyboy.memory[0xDB4D]  # 炸弹数量
        magic_powder = self.pyboy.memory[0xDB4C]  # 魔法粉数量
        secrets = self.pyboy.memory[0xDB0F]  # 秘密贝壳数量
        golden_leaves = self.pyboy.memory[0xDB15]  # 金叶数量
        instruments = [self.pyboy.memory[0xDB65 + i] for i in range(8)]  # 已获得的乐器
        map_progress = [self.pyboy.memory[0xD800 + i] for i in range(256)]  # 已访问的地图区域

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
        self.pyboy.send_input(self.valid_actions[action])
        self.pyboy.tick(self.action_freq)
        # 释放动作
        self.pyboy.send_input(self.release_actions[action])

    def step(self, action):
        # 处理动作
        self._take_action(action)

        # 获取新的状态
        state = self._get_obs()

        # 计算奖励
        reward = self._get_reward()

        # 检查是否结束
        done = self._is_done()

        return state, reward, done, {}

    def get_game_state_reward(self):
        # 自定义奖励函数，基于游戏状态的不同进行奖励计算
        current_health = self.pyboy.memory[0xDB5A]

        # # 修改：假设敌人状态存储在特定内存地址，这里只是示例，需要根据实际情况修改
        # enemy1_dead = self.pyboy.memory[0xXXXX] == 1  # 敌人1死亡标志
        # enemy2_dead = self.pyboy.memory[0xYYYY] == 1  # 敌人2死亡标志
        #
        # # 更新杀敌计数
        # new_enemies_killed = int(enemy1_dead) + int(enemy2_dead)
        # killed_reward = (new_enemies_killed - self.enemies_killed) * 100  # 每杀死一个敌人奖励100
        # self.enemies_killed = new_enemies_killed

        # 修改：假设钥匙获取状态存储在特定内存地址，这里只是示例，需要根据实际情况修改
        key_status = self.pyboy.memory[0xDBD0] == 1
        if key_status and not self.key_obtained:
            key_reward = 100000  # 获得钥匙奖励1000
            self.key_obtained = True
        else:
            key_reward = 0

        # 生命值奖励，生命值降低给予负奖励
        health_reward = (current_health - self.last_health) * 10
        self.last_health = current_health

        # 综合奖励
        total_reward = key_reward + health_reward

        return total_reward

    def bit_count(self, value):
        return bin(value).count('1')

    def read_m(self, addr):
        return self.pyboy.memory[addr]

    def _get_reward(self):
        # 这里可以根据游戏状态计算奖励，例如使用 get_game_state_reward 方法
        return self.get_game_state_reward()

    def _is_done(self):
        # 这里可以根据游戏状态判断是否结束，例如生命值为 0 等
        current_health = self.pyboy.memory[0xDB5A]
        return current_health == 0

    def close(self):
        self.pyboy.stop()

config = {
    "session_path": r"D:\codes\codes_pycharm\da_chuang\sessions",  # 存储路径
    "save_video": True,  # 是否保存视频
    "headless": False,  # 是否无界面模式
    "action_freq": 1,  # 动作频率
    "max_steps": 10000000,  # 最大步骤数
    "init_state": r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb.state"  # 初始化状态文件路径
}

# 使用环境
env = ZeldaEnv(rom_path=r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb", config=config)

# 重置环境并获取初始状态
state, _ = env.reset()

# 模拟智能体的行为
for _ in range(100000):
    action = env.action_space.sample()  # 随机选择一个动作
    state, reward, done, _ = env.step(action)

    # 打印奖励
    print(f"Reward: {reward}")

    if done:
        break

# 关闭环境
env.close()