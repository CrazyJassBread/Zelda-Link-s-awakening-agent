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

while pyboy.tick():
    # 获取精灵坐标
    sprite = pyboy.get_sprite(2)  # 获取第一个精灵（通常是主角）
    x = sprite.x
    y = sprite.y
    print(f"Sprite position: x={x}, y={y}")  # 打印精灵位置
    
pyboy.stop()