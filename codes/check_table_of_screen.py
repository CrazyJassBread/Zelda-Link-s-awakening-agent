from pyboy import PyBoy
import numpy as np
from colorama import init, Fore, Style

#############################
#输出矩阵
############################


# 初始化 colorama
init()

# 定义输出文件路径
output_file = "output.txt"

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

# 设置打印选项，确保完整输出矩阵
np.set_printoptions(threshold=np.inf)

# 初始化上一帧的游戏区域矩阵
prev_game_area_matrix = pyboy.game_area()

# 打开文件以写入结果
with open(output_file, 'w') as f:
    # 主循环，用于人工玩游戏并监控矩阵变化
    while  pyboy.tick():
        # 获取当前游戏区域矩阵
        current_game_area_matrix = pyboy.game_area()

        # 创建一个与矩阵相同形状的布尔矩阵，标记出变化的位置
        changed_mask = current_game_area_matrix != prev_game_area_matrix

        # 格式化矩阵，将变化的元素标记出来
        formatted_matrix = []
        for i in range(current_game_area_matrix.shape[0]):
            row = []
            for j in range(current_game_area_matrix.shape[1]):
                if changed_mask[i, j]:
                    # 用红色标记变化的元素
                    element_str = f"{Fore.RED}{current_game_area_matrix[i, j]}{Style.RESET_ALL}"
                else:
                    element_str = str(current_game_area_matrix[i, j])
                row.append(element_str)
            formatted_matrix.append(" ".join(row))

        # 构建完整的矩阵字符串
        matrix_str = "\n".join(formatted_matrix)

        # 打印并记录当前游戏区域矩阵
        result_str = f"当前游戏区域矩阵（帧号: {pyboy.frame_count}）:\n{matrix_str}\n"
        print(result_str)
        f.write(result_str.replace(Fore.RED, "").replace(Style.RESET_ALL, ""))

        # 更新上一帧的游戏区域矩阵
        prev_game_area_matrix = current_game_area_matrix

# 关闭模拟器
pyboy.stop()