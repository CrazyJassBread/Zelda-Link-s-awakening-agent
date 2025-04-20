import gym
from gym import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from pathlib import Path
import time
# #################################################################################
# 找到了自己的(x, y), 和钥匙的位置，增加奖励函数的方法, 位置学错了，直接跑过去了，不*2了 #
###################################################################################

class ZeldaEnv(gym.Env):
    def __init__(self, rom_path, config=None):
        super(ZeldaEnv, self).__init__()

        # 配置初始化
        self.start_time = time.time()

        self.s_path = Path(config.get("session_path", "./"))  # 存储路径
        self.print_rewards = config.get("print_rewards", False)  # 是否打印奖励
        self.headless = config["headless"]  # 是否无头模式（不显示图形界面）
        self.init_state = config["init_state"]  # 初始状态文件路径
        # self.save_video = config["save_video"]  # 是否保存视频
        # self.fast_video = config.get("fast_video", False)  # 是否快速视频
        self.frame_stacks = 3  # 堆叠的帧数
        self.action_freq = config.get("action_freq", 4)  # 动作频率
        self.max_steps = config.get("max_steps", 10000000)  # 最大步骤数

        self.s_path.mkdir(exist_ok=True)  # 创建存储路径
        # self.full_frame_writer = None  # 完整视频帧写入器
        # self.model_frame_writer = None  # 模型视频帧写入器
        # self.map_frame_writer = None  # 地图视频帧写入器

        self.target_x = 33  # 钥匙的x坐标
        self.target_y = 44  # 钥匙的y坐标
        self.last_x = None  # 用于记录上一次的x坐标
        self.last_y = None  # 用于记录上一次的y坐标
        self.visited_positions = set()  # 用于记录已访问的位置

        self.health_history = []
        self.health_stable_threshold = 3  # 健康值保持不变的阈值步数
       
    
        self.reset_count = 0  # 重置计数器
        # 定义动作空间（允许的按钮按下动作）
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            # WindowEvent.PRESS_BUTTON_START,
        ]

        # 定义释放动作
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            # WindowEvent.RELEASE_BUTTON_START
        ]

        # 设置动作空间（动作数量与 valid_actions 一致）
        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.metadata = {"render.modes": []}  # 渲染模式
        self.reward_range = (0, 15000000)  # 奖励范围
        self.cumulative_reward = 0  # 累计奖励初始化

        # 定义状态空间
        self.observation_space = spaces.Dict(
            {
                "health": spaces.Box(low=0, high=14, shape=(1,), dtype=np.uint8),  # 当前生命值（0-14颗心）
                "max_health": spaces.Box(low=0, high=14, shape=(1,), dtype=np.uint8),  # 最大生命值（最大14颗心）
                "rupees": spaces.Box(low=0, high=999, shape=(1,), dtype=np.uint16),  # 当前卢比数量
                "position": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),  # 当前坐标（位置）
                "inventory": spaces.MultiBinary(10),  # 库存物品（最多 10 个物品）
                "key_count": spaces.Box(low=0, high=99, shape=(1,), dtype=np.uint8),  # 钥匙数量
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
        # self.enemies_killed = 0
        self.key_obtained = False
        self.last_health = 1

         # 添加记录获取钥匙时间的列表和文件路径
        self.key_obtained_times = []
        self.results_file = Path(self.s_path) / "key_obtained_results_v7.txt"
        self.episode_start_time = None


    def reset(self, seed=None, options={}):
        # 记录每个回合的开始时间
        self.episode_start_time = time.time()

        self.health_history = []

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
        # self.base_event_flags = sum([
        #     self.bit_count(self.read_m(i))
        #     for i in range(0xD747, 0xD87E)  # 假设事件标志的地址范围
        # ])

        self.current_event_flags_set = {}

        # 初始化进度奖励
        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()  # 现在是整数
        self.total_reward = self.progress_reward  # 直接赋值
        self.reset_count += 1

      
        self.key_obtained = False
        self.visited_positions.clear()  # 使用 clear() 方法清空集合
        # 重置坐标
        self.last_x = None
        self.last_y = None
        self.cumulative_reward = 0

        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}
    

    def get_game_state_reward(self, action=None):
        # 获取精灵当前位置
        sprite = self.pyboy.get_sprite(2)  # 使用精灵索引2
        current_x = sprite.x
        current_y = sprite.y
        
        # 初始化各种奖励
        position_reward = 0
        exploration_reward = 0
        movement_reward = 0
        area_reward = 0  # 新增区域奖励
        
        # 检查是否在指定区域内 -- 感觉莫名其妙的靠近边界
        if (24 <= current_x <= 119 and 22 <= current_y <= 36):  # 区域2.1
            area_reward += 10
            # print("In area 2.1")
        elif (0 <= current_x <= 144 and 36 <= current_y <= 65):  # 区域2.2
            area_reward += 10
            # print("In area 2.2")
        elif (24 <= current_x <= 119 and 65 <= current_y <= 81):  # 区域2.3
            area_reward += 10
            # print("In area 2.3")
        else:
            area_reward -= 50  # 不在指定区域内给予负奖励
            # print("Outside of target areas")
        
        # 计算与目标位置的距离
        print(f"Current position: ({current_x}, {current_y})")
        print(f"Target position: ({self.target_x}, {self.target_y})")
        current_distance = np.sqrt((current_x - self.target_x)**2 + (current_y - self.target_y)**2)
        
        # 如果是第一次运行，初始化last坐标
        if self.last_x is None:
            self.last_x = current_x
            self.last_y = current_y
            position_reward = 0
        else:
            # 计算与上一次位置的距离
            last_distance = np.sqrt((self.last_x - self.target_x)**2 + (self.last_y - self.target_y)**2)
            
            distance_improvement = last_distance - current_distance
            position_reward = distance_improvement * 50

        # 2. 添加阶段性奖励
        if current_distance < 10:
            position_reward += 300
        elif current_distance < 20:
            position_reward += 200
        elif current_distance < 30:
            position_reward += 100
                
        # 探索新区域的奖励
        position_tuple = (current_x, current_y)
        if position_tuple not in self.visited_positions:
            exploration_reward = 30  # 访问新位置给予奖励
            self.visited_positions.add(position_tuple)
        
        # 更新上一次的位置
        self.last_x = current_x
        self.last_y = current_y
        
        # 获取原有的奖励计算
        current_health = self.pyboy.memory[0xDB5A]
        current_position = self.pyboy.memory[0xDBAE]
        
        # 检查是否在正确的房间
        if current_position == 0x3a:
            room_reward = 200
        else:
            room_reward = -100
        
        # 钥匙奖励
        key_status = self.pyboy.memory[0xDBD0] == 1
        if key_status and not self.key_obtained:
            key_reward = 1000
            self.key_obtained = True
            # 计算本次获得钥匙所用时间
            episode_time = time.time() - self.episode_start_time
            total_time = time.time() - self.start_time
            self.key_obtained_times.append({
                'episode_time': episode_time,
                'total_time': total_time,
                'step_count': self.step_count
            })
            
            # 将结果写入文件
            with open(self.results_file, 'a') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Key obtained at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Episode time: {episode_time:.2f} seconds\n")
                f.write(f"Total training time: {total_time:.2f} seconds\n")
                f.write(f"Steps taken: {self.step_count}\n")
                f.write(f"Reset count: {self.reset_count}\n")
                f.write(f"{'='*50}\n")
            
            # 打印当前结果
            print("\n" + "="*50)
            print(f"Congratulations! Key obtained!")
            print(f"Episode time: {episode_time:.2f} seconds")
            print(f"Total training time: {total_time:.2f} seconds")
            print(f"Steps taken: {self.step_count}")
            print(f"Results saved to: {self.results_file}")
            print("="*50 + "\n")
        else:
            key_reward = 0
        
        # 生命值奖励
        health_reward = (current_health - self.last_health) * 100
        # print(f"health: {health_reward}")
        self.last_health = current_health
        
        # 打印调试信息
        print(f"Position: ({current_x}, {current_y}), Distance to target: {current_distance:.2f}")
        print(f"Position reward: {position_reward}, Exploration reward: {exploration_reward}")
        print(f"Area reward: {area_reward}")  # 新增区域奖励的打印

        # 更新健康值历史
        self.health_history.append(current_health)
        if len(self.health_history) > self.health_stable_threshold:
            self.health_history.pop(0)
        
        # 检查健康值是否保持稳定
        health_stable = False
        if len(self.health_history) == self.health_stable_threshold:
            if all(h == self.health_history[0] for h in self.health_history):
                health_stable = True
        
        # 如果健康值稳定且使用了A或B键，给予负奖励
        button_penalty = 0
        if health_stable:
            if self.valid_actions[action] in [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B]:
                button_penalty = -50  # 设置惩罚值
        
        # 综合所有奖励
        total_reward = (
            position_reward +  # 接近目标的奖励
            # exploration_reward +  # 探索新区域的奖励
            movement_reward +  # 移动奖励
            room_reward +  # 房间位置奖励
            key_reward +  # 获得钥匙奖励
            health_reward +  # 生命值奖励
            area_reward  +# 新增区域奖励
            button_penalty  # 添加按键惩罚
        )
        
        self.cumulative_reward += total_reward
        return total_reward
    

    def bit_count(self, value):
        return bin(value).count('1')

    def read_m(self, addr):
        return self.pyboy.memory[addr]

    def _get_reward(self,action = None):
        # 这里可以根据游戏状态计算奖励，例如使用 get_game_state_reward 方法
        return self.get_game_state_reward(action)

    def _is_done(self):
        # 这里可以根据游戏状态判断是否结束，例如生命值为 0 等
        # current_health = self.pyboy.memory[0xDB5A]
        return self.key_obtained == True

    # def close(self):
    #     self.pyboy.stop()
    def close(self):
        # 在环境关闭时打印汇总信息
        if self.key_obtained_times:
            print("\nTraining Summary:")
            print(f"Total successful attempts: {len(self.key_obtained_times)}")
            avg_episode_time = sum(x['episode_time'] for x in self.key_obtained_times) / len(self.key_obtained_times)
            avg_steps = sum(x['step_count'] for x in self.key_obtained_times) / len(self.key_obtained_times)
            print(f"Average time per successful attempt: {avg_episode_time:.2f} seconds")
            print(f"Average steps per successful attempt: {avg_steps:.2f}")
            
            # 将汇总信息也写入文件
            with open(self.results_file, 'a') as f:
                f.write("\nTraining Summary:\n")
                f.write(f"Total successful attempts: {len(self.key_obtained_times)}\n")
                f.write(f"Average time per successful attempt: {avg_episode_time:.2f} seconds\n")
                f.write(f"Average steps per successful attempt: {avg_steps:.2f}\n")
                
        self.pyboy.stop()


    def _get_obs(self):
        # 获取当前状态
        current_health = self.pyboy.memory[0xDB5A]  # 当前生命值
        max_health = self.pyboy.memory[0xDB5B]  # 最大生命值
        rupees = self.pyboy.memory[0xDB5D] * 256 + self.pyboy.memory[0xDB5E]  # 当前卢比数量
        position = self.pyboy.memory[0xDBAE]  # 当前在地牢中的位置

        # 其他游戏状态项
        inventory = [self.pyboy.memory[0xDB02 + i] for i in range(10)]  # 库存物品
        key_count = self.pyboy.memory[0xDBD0]  # 钥匙数量
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
            "key_count": np.array([key_count]),
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

        # 监控敌人内存变化
        # current_enemy_memory = np.array([self.pyboy.memory[i] for i in range(0xD700, 0xD79C)])
        # changed_indices = np.where(current_enemy_memory != self.prev_enemy_memory)[0]
        # if len(changed_indices) > 0:
        #     print(f"Enemy memory changed at addresses: {[hex(0xD700 + i) for i in changed_indices]}")
        # self.prev_enemy_memory = current_enemy_memory

    def step(self, action):
        # 处理动作
        self._take_action(action)

        # 获取新的状态
        state = self._get_obs()

        # 获取当前位置
        current_position = self.pyboy.memory[0xDBAE]

        # 检查位置是否不等于 0x3a
        if current_position != 0x3a:
            print("Position is not 0x3a, restarting the game!")
            self.pyboy.send_input(WindowEvent.STATE_LOAD)  # 按下 'x' 键
            self.pyboy.tick()  # 等待一帧以确保输入被处理
            self.pyboy.send_input(WindowEvent.STATE_LOAD)  # 释放 'x' 键
            state = self.reset()[0]  # 重置环境并获取初始状态

        # 检查生命值
        current_health = self.pyboy.memory[0xDB5A]  # 当前生命值
        if current_health == 0:
            # 生命值为零，按下 'x' 键重新开始
            print("Health is zero, restarting the game!")
            self.pyboy.send_input(WindowEvent.STATE_LOAD)  # 按下 'x' 键
            self.pyboy.tick()  # 等待一帧以确保输入被处理
            self.pyboy.send_input(WindowEvent.STATE_LOAD)  # 释放 'x' 键
            state = self.reset()[0]  # 重置环境并获取初始状态

        # # 根据动作类型调整奖励, 这里是根据钥匙的位置来的，钥匙偏左，并且稍微偏上（训练过程总是向下走不知道为什么）
        # if self.valid_actions[action] == WindowEvent.PRESS_ARROW_LEFT:
        #     additional_reward = 50  # 向左走加10点奖励
        # elif self.valid_actions[action] == WindowEvent.PRESS_ARROW_RIGHT:
        #     additional_reward = -50  # 向右走减10点奖励
        # elif self.valid_actions[action] == WindowEvent.PRESS_ARROW_UP:
        #     additional_reward = 10
        # else:
        #     additional_reward = 0

            # 计算奖励
        reward = self._get_reward(action)

        # 检查是否结束
        done = self._is_done()

        # 检查是否结束
        terminated = self._is_done()
        truncated = False  # 如果有其他终止条件，可以在此设置 truncated 为 True

        info = {
            "TimeLimit.terminated": terminated,
            "TimeLimit.truncated": truncated
        }

        return state, reward, terminated, truncated, info
    

    

config = {
    "session_path": r"D:\codes\codes_pycharm\da_chuang\sessions",  # 存储路径
    "save_video": True,  # 是否保存视频
    "headless": False,  # 是否无界面模式
    "action_freq": 1,  # 动作频率
    "max_steps": 200000,  # 最大步骤数
    "init_state": r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb.state"  # 初始化状态文件路径
}


from stable_baselines3 import PPO
# from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.policies import MultiInputPolicy
import gym

# 使用环境
env = ZeldaEnv(rom_path=r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb", config=config)
# 将环境封装到一个 vectorized 环境中（Stable Baselines3 需要封装的环境）
# env = DummyVecEnv([lambda: env])  # 将环境包装成一个向量化环境，这对于并行训练非常有用

# 创建 PPO 模型
model = PPO("MultiInputPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=1000000)  # 您可以根据需要调整 timesteps

# 保存模型

# 测试模型（根据训练的模型进行测试）
obs = env.reset()
for i in range(200000):  # 运行1000步
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()  # 如果您希望查看训练过程中的结果，可以启用渲染
    if dones.any():  # 如果游戏结束
        obs = env.reset() # 如果游戏结束，重置环境并获取新的状态
        break

# 关闭环境
env.close()