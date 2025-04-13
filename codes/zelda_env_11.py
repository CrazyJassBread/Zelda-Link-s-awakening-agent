"""
这里的版本添加了不同的算法进行比较，并且对于之前的代码进行了一些些的优化
"""

import gym
from gym import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from pathlib import Path
import time

class ZeldaEnv(gym.Env):
    def __init__(self, rom_path, config=None):
        super(ZeldaEnv, self).__init__()
        '''
        初始化变量
        '''
        self.s_path = Path(config.get("session_path", "./"))  # 存储路径
        self.headless = config["headless"]  # 是否无头模式 (不显示图形界面)
        self.init_state = config["init_state"]
        self.frame_stacks = 3  # 堆叠的连续游戏帧数， 通过将多个连续帧组合在一起，AI可以推断出物体移动方向和速度等动态信息
        self.action_freq = config.get("action_freq", 4)  # 控制每个AI决策后执行多少个游戏帧
        self.max_steps = config.get("max_steps", 1e7)  # 单个训练回合的最大步数限制， 防止AI陷入死循环或无法完成任务的情况

        self.s_path.mkdir(exist_ok=True)  # 文件路径已经存在就不会报错

        '''
        奖励函数里面的变脸初始化
        '''
        self.key_x = 38
        self.key_y = 43
        self.last_x = None
        self.last_y = None

        self.visited_positions = set()  # 用于记录已经访问过的位置

        self.health_history = []
        self.health_stable_threshold = 3  # 健康值保持不变的阈值步数 ，这里保持不变是为了检查敌人是否还存在，这里是否有这个必要留存呢

        self.reset_count = 0  # 记录AI需要多少次尝试（重置）才能成功完成任务

        self.enemies_killed = 0
        self.key_obtained = False
        self.last_health = 1
        self.enemy_phase = True
        self.last_enemy_count = 2

        # 定义动作空间(允许按钮按下的动作空间)
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP, 
            WindowEvent.PRESS_BUTTON_A, 
            WindowEvent.PRESS_BUTTON_B
        ]
        # 定义释放动作
        self.release_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP, 
            WindowEvent.PRESS_BUTTON_A, 
            WindowEvent.PRESS_BUTTON_B
        ]

        self.action_space = spaces.Discrete(len(self.valid_actions))

        head = "null" if config["headless"] else "SDL2"

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)  #设置模拟器的仿真速度，使得人眼能够跟上

        self.pyboy = PyBoy(
            rom_path,
            window=head
        )

    def reset(self, seed = None, options={}):
        super.reset(seed=seed)
        self.last_x = None
        self.last_y = None

        self.visited_positions = set()  # 用于记录已经访问过的位置

        self.health_history = []
        self.health_stable_threshold = 3  # 健康值保持不变的阈值步数 ，这里保持不变是为了检查敌人是否还存在，这里是否有这个必要留存呢

        self.reset_count = 0  # 记录AI需要多少次尝试（重置）才能成功完成任务

        self.enemies_killed = 0
        self.key_obtained = False
        self.last_health = 1
        self.enemy_phase = True
        self.last_enemy_count = 2




  





        

        


