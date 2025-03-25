from pyboy import PyBoy
from pyboy.utils import WindowEvent
import numpy as np

# 定义游戏 ROM 文件路径
rom_path = r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb"
# 定义初始游戏状态文件（.state 文件）的路径
init_state_path = r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb.state"

# 初始化 PyBoy 模拟器
pyboy = PyBoy(rom_path, window="SDL2")

# 尝试加载初始游戏状态
try:
    with open(init_state_path, "rb") as f:
        # 加载保存的游戏状态
        pyboy.load_state(f)
        print(f"Successfully loaded game state from {init_state_path}")
except FileNotFoundError:
    print(f"Warning: Game state file {init_state_path} not found, starting a new game.")

# # 初始化敌人内存监控
# prev_enemy_memory = np.array([pyboy.memory[i] for i in range(0xD700, 0xD79C)])

# 主循环，用于人工玩游戏并监控内存变化
while  pyboy.tick():
    # # 查看 DBAE 地址的数值
    DBD0_value = pyboy.memory[0xDBD0]
    # DB01_value = pyboy.memory[0xDB01]
    print(f"Value at address DB00_value: {hex(DBD0_value)},")
          # f" Value at address DB01_value: {hex(DB01_value)}")

# 关闭模拟器
pyboy.stop()