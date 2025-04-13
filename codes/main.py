from pyboy import PyBoy
from pyboy.utils import WindowEvent
import numpy as np
from datetime import datetime
import os

# 定义游戏 ROM 文件路径
rom_path = r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb"
# 定义初始游戏状态文件（.state 文件）的路径
init_state_path = r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb.state"

def get_enemy_count(self):
    """获取当前场景中的敌人数量"""
    visible_sprites_count = 0
    try:
        for idx in range(4, 40):  # 从4开始到39的精灵
            if idx == 2 or idx == 3:  # 跳过编号为2和3的精灵
                continue
            sprite = self.pyboy.get_sprite(idx)
            if sprite.on_screen:
                visible_sprites_count += 1
        
        # 计算敌人数量: (可见精灵数-2)/2
        enemy_count = max(0, (visible_sprites_count) // 2) + 1
        print(f"enemy_count : {enemy_count}")
        return enemy_count
    except Exception as e:
        print(f"获取敌人数量出错: {e}")
        return 0

# 创建日志文件，文件名包含当前时间
# 确保log文件夹存在
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = os.path.join(log_dir, f"zelda_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# 定义一个同时输出到控制台和文件的函数
def log_message(message, file):
    print(message)
    file.write(message + "\n")

# 初始化 PyBoy 模拟器
pyboy = PyBoy(rom_path, window="SDL2")

# 打开日志文件
with open(log_file_path, "w", encoding="utf-8") as log_file:
    # 记录开始时间
    start_time = datetime.now()
    log_message(f"开始记录: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", log_file)

    # 尝试加载初始游戏状态
    try:
        with open(init_state_path, "rb") as f:
            # 加载保存的游戏状态
            pyboy.load_state(f)
            log_message(f"Successfully loaded game state from {init_state_path}", log_file)
    except FileNotFoundError:
        log_message(f"Warning: Game state file {init_state_path} not found, starting a new game.", log_file)

    frame_count = 0
    max_frames = 10000000000  # 可以根据需要调整，防止无限循环

    while pyboy.tick() and frame_count < max_frames:
        # 获取精灵坐标
        sprite = pyboy.get_sprite(3)  # 获取第一个精灵（通常是主角）
        is_alive = sprite.on_screen
        
        log_message("---------------------------------------------------------------------", log_file)
        
        # 读取所有需要记录的sprite (0-39的所有sprite)
        # sprite_indices = list(range(40))  # 生成0到39的列表
        
        # for idx in sprite_indices:
        #     try:
        #         sprite = pyboy.get_sprite(idx)
                
        #         # 只输出on_screen为True的精灵
        #         if sprite.on_screen:
        #             sprite_info = f"Sprite [{idx}]: Position: ({sprite.x}, {sprite.y}),Tiles: (Tile: {sprite.tiles[0]}, Tile: {sprite.tiles[1]}), On screen: {sprite.on_screen}"
        #             log_message(sprite_info, log_file)
        #     except Exception as e:
        #         log_message(f"错误读取Sprite {idx}: {str(e)}", log_file)
        # visible_sprites_count = 0
        # try:
        #     for idx in range(13, 23):  # 从4开始到39的精灵
        #         if idx == 2 or idx == 3:  # 跳过编号为2和3的精灵
        #             continue
        #         sprite = pyboy.get_sprite(idx)
        #         # if sprite.on_screen:
        #         #     visible_sprites_count += 1
        #         #     sprite_info = f"Sprite [{idx}]: Position: ({sprite.x}, {sprite.y}),Tiles: (Tile: {sprite.tiles[0]}, Tile: {sprite.tiles[1]}), On screen: {sprite.on_screen}"
        #         #     log_message(sprite_info, log_file)
        #         sprite_info = f"Sprite [{idx}]: Position: ({sprite.x}, {sprite.y}),Tiles: (Tile: {sprite.tiles[0]}, Tile: {sprite.tiles[1]}), On screen: {sprite.on_screen}"
        #         log_message(sprite_info, log_file)


                
            
        #     # 计算敌人数量: (可见精灵数-2)/2
        #     enemy_count = max(0, (visible_sprites_count - 2) // 2)
        #     print(f"visible_sprites_count : {visible_sprites_count}, enemy_count : {enemy_count}")
        # except Exception as e:
        #     print(f"获取敌人数量出错: {e}")
                    
        # 原有代码部分 - 保留这部分逻辑
        # sprite_enemy_15 = pyboy.get_sprite(13)
        # sprite_enemy_16 = pyboy.get_sprite(14)
        
        # x_enemy_15, y_enemy_15 = sprite_enemy_15.x, sprite_enemy_15.y
        # x_enemy_16, y_enemy_16 = sprite_enemy_16.x, sprite_enemy_16.y

        # position_info = f"Frame {frame_count}: position : 13 :( {x_enemy_15},  {y_enemy_15} ) , 14 : ( {x_enemy_16}, {y_enemy_16} )"
        # log_message(position_info, log_file)
        sprite_enemy_2 = pyboy.get_sprite(2)
        sprite_enemy_3 = pyboy.get_sprite(3)
        
        x_enemy_2, y_enemy_2 = sprite_enemy_2.x, sprite_enemy_2.y
        x_enemy_3, y_enemy_3 = sprite_enemy_3.x, sprite_enemy_3.y

        position_info = f"Frame {frame_count}: position : 13 :( {x_enemy_2},  {y_enemy_2} ) , 14 : ( {x_enemy_3}, {y_enemy_3} )"
        log_message(position_info, log_file)
        
        # 记录主角状态
        # character_info = f"主角在屏幕上: {is_alive}, 位置: ({sprite.x}, {sprite.y})"
        # log_message(character_info, log_file)
        
        frame_count += 1

    # 记录结束时间和总帧数
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    log_message(f"\n记录结束: {end_time.strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_message(f"总帧数: {frame_count}", log_file)
    log_message(f"运行时间: {duration:.2f} 秒", log_file)

# 关闭PyBoy
pyboy.stop()
print(f"日志已保存到: {log_file_path}")