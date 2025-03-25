from pyboy import PyBoy
from pyboy.utils import WindowEvent
import numpy as np

#############################
# 输出变化坐标
#############################



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

# 初始化敌人内存监控
# prev = np.array([pyboy.memory[i] for i in range(0xD800, 0xD8FF)])

# 初始化游戏区域矩阵监控
prev_game_area_matrix = pyboy.game_area()

# 设置打印选项，确保完整输出矩阵
np.set_printoptions(threshold=np.inf)

# 打开一个文本文件用于存储结果
with open('output.txt', 'a') as f:
    # 主循环，用于人工玩游戏并监控内存变化
    while pyboy.tick():
        # 获取当前游戏区域矩阵
        current_game_area_matrix = pyboy.game_area()

        # 比较当前和上一次的游戏区域矩阵，找出变化的位置
        changed_positions = np.argwhere(current_game_area_matrix != prev_game_area_matrix)

        if changed_positions.size > 0:
            # 打印并记录矩阵变化信息
            matrix_change_info = "游戏区域矩阵发生变化的位置及值:\n"
            print(matrix_change_info)
            f.write(matrix_change_info)
            for pos in changed_positions:
                row, col = pos
                old_value = prev_game_area_matrix[row, col]
                new_value = current_game_area_matrix[row, col]
                position_change_info = f"位置: ({row}, {col}), 旧值: {old_value}, 新值: {new_value}\n"
                print(position_change_info)
                f.write(position_change_info)

        # 更新上一次的游戏区域矩阵
        prev_game_area_matrix = current_game_area_matrix

        # 记录当前游戏区域矩阵
        matrix_info = "当前游戏区域矩阵:\n" + str(current_game_area_matrix) + "\n"
        f.write(matrix_info)


# 关闭模拟器
pyboy.stop()