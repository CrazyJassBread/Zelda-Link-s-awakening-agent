"""
只是修改了奖励函数，把杀敌人的动作修改成自己学习出来的，因为整体是按照宝可梦的代码来的，所以整体上修改了代码，看上去更加清晰，更加符合当前的游戏。
"""

import gym
from gym import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from pathlib import Path
import time
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv




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

        # self.health_history = []
        self.health_stable_threshold = 3  # 健康值保持不变的阈值步数 ，这里保持不变是为了检查敌人是否还存在，这里是否有这个必要留存呢

        # self.reset_count = 0  # 记录AI需要多少次尝试（重置）才能成功完成任务

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
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP, 
            WindowEvent.RELEASE_BUTTON_A, 
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        # 定义观察空间
        self.observation_space = spaces.Dict({
            "health": spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
            "position": spaces.Box(low=0, high=255, shape=(1,), dtype=np.float32)
        })

        head = "null" if config["headless"] else "SDL2"

        # if not config["headless"]:
        #     self.pyboy.set_emulation_speed(6)  #设置模拟器的仿真速度，使得人眼能够跟上

        self.pyboy = PyBoy(
            rom_path,
            window=head
        )
        
        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)  # 设置模拟器的仿真速度


    def reset(self, seed = None, options={}):
        self.last_x = None
        self.last_y = None

        self.visited_positions.clear()  # 用于记录已经访问过的位置

        # self.health_history = []
        self.enemies_killed = 0
        self.key_obtained = False
        self.last_health = 1
        self.enemy_phase = True
        self.last_enemy_count = 2

        self.seed = seed
        return self._get_obs(), {}
    
    def get_reward(self, action=None):

        sprite_zelda = self.pyboy.get_sprite(2)
        current_zelda_x, current_zelda_y = sprite_zelda.x, sprite_zelda.y

        position_reward = 0
        exploration_reward = 0
        movement_reward = 0
        board_reward = 0  # 不是陷阱的地板
        room_reward = 0
        key_reward = 0
        enemy_reward = 0
        keyboard_reward = 0
        
        # 检查是否在指定的地板上，分为三个区域，掉进陷阱就减奖励
        if (24 <= current_zelda_x <= 119 and 22 <= current_zelda_y <= 36):  # 区域2.1
            board_reward += 10
        elif (0 <= current_zelda_x <= 144 and 36 <= current_zelda_y <= 65):  # 区域2.2
            board_reward += 10
        elif (24 <= current_zelda_x <= 119 and 65 <= current_zelda_y <= 81):  # 区域2.3
            board_reward += 10
        else:
            board_reward -= 50  # 不在指定区域内给予负奖励

        # 获取当前敌人的数量
        current_enemy_count = self.get_enemy_count()
        enemies_killed = self.last_enemy_count - current_enemy_count
        # print(f"current_enemy_count :  {current_enemy_count}, enemies_killed : {enemies_killed}")

        # 处理敌人奖励
        if enemies_killed > 0:
            if enemies_killed == 1:
                enemy_reward = 300  # 提高杀敌奖励
            elif enemies_killed == 2:
                enemy_reward = 700  # 提高杀敌奖励
            # print(f"杀死了{enemies_killed}个敌人! 奖励:{enemy_reward}")

        # 更新敌人阶段状态
        if current_enemy_count == 0:
                self.enemy_phase = False
                # print("所有敌人已清除，进入探索阶段")
        elif current_enemy_count > 1:
            # 如果仍有两个敌人，给予轻微惩罚以鼓励战斗
            enemy_reward = -10
            self.enemy_phase = True

        # 更新敌人数量记录
        self.last_enemy_count = current_enemy_count

        # 按键奖励逻辑
        enemy_nearby = self.is_enemy_nearby(threshold=40)  # 增大检测范围

        # 处理敌人阶段的按键选择
        if self.enemy_phase:
            # 在敌人阶段
            if enemy_nearby:
                # 敌人在附近，使用攻击键(A或B)应该获得正向奖励
                if self.valid_actions[action] in [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B]:
                    keyboard_reward += 350
                    # print("敌人附近使用了攻击键，给予奖励")
                else:
                    # 敌人靠近但没有攻击，适度惩罚
                    keyboard_reward -= 100
                    # print("敌人附近未使用攻击键，给予惩罚")
            else:
                # 敌人不在附近，鼓励移动寻找敌人，使用方向键给少量奖励
                if self.valid_actions[action] in [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT, 
                                              WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_ARROW_UP]:
                    keyboard_reward += 50
                    # print("敌人阶段正在移动寻找敌人")
        else:
            # 在探索阶段，应该移动去获取钥匙
            if self.valid_actions[action] in [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B]:
                # 没有敌人时使用攻击键，轻微惩罚
                keyboard_reward -= 50
                # print("探索阶段不需要攻击，给予轻微惩罚")
            else:
                # 使用方向键移动
                keyboard_reward += 100
                # print("探索阶段使用方向键移动，给予奖励")

        # 计算与钥匙的距离
        current_distance = np.sqrt((current_zelda_x - self.key_x)**2 + (current_zelda_y - self.key_y)**2)
        
        # 对于距离的阶段进行分级增加奖励
        if current_distance < 10:
            position_reward += 300
        elif current_distance < 20:
            position_reward += 200
        elif current_distance < 30:
            position_reward += 100
        
        # 离钥匙近的奖励
        if self.last_x is None:
            self.last_x, self.last_y = current_zelda_x, current_zelda_y
            position_reward = 0
        else:
            last_distance = np.sqrt((self.last_x - self.key_x)**2 + (self.last_y - self.key_y)**2)
            distance_improvement = last_distance - current_distance
            position_reward = distance_improvement * 100
        
        # 探索新区域的奖励
        position_tuple = (current_zelda_x, current_zelda_y)
        if position_tuple not in self.visited_positions:
            exploration_reward += 30
            self.visited_positions.add(position_tuple)

        # 更新上一次的位置
        self.last_x, self.last_y = current_zelda_x, current_zelda_y

        current_health = self.pyboy.memory[0xDB5A]
        health_reward = (current_health - self.last_health) * 100
        self.last_health = current_health

        current_position_ = self.pyboy.memory[0xDBAE]  # 在 8 * 8网格地图里面
        if current_position_ != 0x3a:  # 检查是否在正确的房间里面
            room_reward = -100

        key_status = self.pyboy.memory[0xDBD0] == 1

        if key_status and not self.key_obtained:
            key_reward = 1000
            self.key_obtained = True
            print(f"Congratulations! Key obtained!")

        # 调整奖励权重，增加敌人和攻击奖励的权重
        total_reward = (
            position_reward * 0.8 +      # 减少位置奖励的权重
            exploration_reward + 
            movement_reward +
            room_reward + 
            key_reward +
            health_reward + 
            board_reward + 
            enemy_reward * 1.5 +         # 增加敌人奖励的权重
            keyboard_reward * 1.2        # 增加按键奖励的权重
        )

        # print(f"总奖励: {total_reward}, 敌人数量: {current_enemy_count}, 敌人阶段: {'是' if self.enemy_phase else '否'}")
        return total_reward
    
    def _is_done(self):
        return self.key_obtained == True
    
    def close(self):
        self.pyboy.stop()

    def _get_obs(self):
        current_health = self.pyboy.memory[0xDB5A]
        position_ = self.pyboy.memory[0xDBAE] # 8 *  8 地牢网格

        state = {
            "health" : np.array([current_health], dtype=np.float32),
            "position" : np.array([position_], dtype=np.float32),
        }
        return state
    
    def _take_action(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        self.pyboy.tick(self.action_freq)
        self.pyboy.send_input(self.release_actions[action])

    def step(self, action):
        original_action = action

        self._take_action(action)
        state = self._get_obs()
        current_health = self.pyboy.memory[0xDB5A]
        self.last_health = current_health

        current_pos_8 = self.pyboy.memory[0xDBAE]

        if current_pos_8 != 0x3a or current_health == 0:
            self.pyboy.send_input(WindowEvent.STATE_LOAD)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.STATE_LOAD)
            state = self.reset()[0]

        reward = self.get_reward(action=action)
        terminated = self._is_done()
        truncated = False

        info = {
            "TimeLimit.terminated": terminated,
            "TimeLimit.truncated": truncated,
        }
        return state, reward, terminated, truncated, info       

    def get_enemy_count(self):
        visible_sprites_count = 0

        try:
            for idx in  range(13, 23):
                sprite = self.pyboy.get_sprite(idx)
                if sprite.on_screen:
                    visible_sprites_count += 1
            
            # 计算敌人数量 
            enemy_count = max(0, (visible_sprites_count + 1) // 2)
            return enemy_count
        except Exception as e:
            print(f"获取敌人数量出错: {e}")
            return 0
    
    def get_enemy_positions(self):
        enemy_positions = []
        try:
            for idx in  range(13, 23):
                sprite = self.pyboy.get_sprite(idx)
                if sprite.on_screen:
                    # print(f"Sprite [{idx}]: Position: ({sprite.x}, {sprite.y}),Tiles: (Tile: {sprite.tiles[0]}, Tile: {sprite.tiles[1]}), On screen: {sprite.on_screen}")
                    enemy_positions.append((sprite.x, sprite.y))
            return enemy_positions
        except Exception as e:
            print(f"获取敌人数量出错: {e}")
            return 0
        
    def is_enemy_nearby(self, threshold=40):
        zelda_sprite = self.pyboy.get_sprite(2)
        zelda_sprite_x, zelda_sprite_y = zelda_sprite.x, zelda_sprite.y

        enemy_positions = self.get_enemy_positions()

        for enemy_x, enemy_y in  enemy_positions:
            dis = np.sqrt((zelda_sprite_x - enemy_x)**2 + (zelda_sprite_y - enemy_y)**2)
            if dis < threshold:
                return True
            
        return False



from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import gym

config = {
    "session_path": r"D:\codes\codes_pycharm\da_chuang\sessions",  # 存储路径
    "headless": False,  # 是否无界面模式
    "action_freq": 1,  # 动作频率
    "max_steps": 200000,  # 最大步骤数
    "init_state": r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb.state"  # 初始化状态文件路径
}

env = ZeldaEnv(rom_path=r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb", config=config)

model = PPO("MultiInputPolicy", env, verbose=1)

# 设置模型保存路径
model_save_dir = Path(config["session_path"]) / "zelda_models"
model_save_dir.mkdir(exist_ok=True)
print(f"模型将保存在: {model_save_dir}")

# 定义回调函数，用于定期保存模型
from stable_baselines3.common.callbacks import CheckpointCallback
# 每10000步保存一次模型
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=str(model_save_dir),
    name_prefix="zelda_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# 训练模型
model.learn(total_timesteps=1000000, callback=checkpoint_callback)  # 使用回调函数定期保存

# 训练完成后保存最终模型
final_model_path = model_save_dir / "zelda_final_model"
model.save(final_model_path)
print(f"最终模型已保存至: {final_model_path}")

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
