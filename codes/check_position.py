from pyboy import PyBoy
import os
import numpy as np  # 新增numpy库导入


def analyze_positions(rom_path, save_state_path=None):
    if not os.path.exists(rom_path):
        raise FileNotFoundError(f"ROM文件未找到: {rom_path}")

    with PyBoy(rom_path) as pyboy:
        if save_state_path and os.path.exists(save_state_path):
            with open(save_state_path, "rb") as f:
                pyboy.load_state(f)

        for _ in range(30):
            pyboy.tick()

        # 获取游戏区域
        game_area = pyboy.game_area()
        
        # 获取所有唯一的图块值
        unique_elements = np.unique(game_area).tolist()
        
        print("游戏区域形状:", game_area.shape)
        print("\n分析图块信息:")
        
        # 创建一个字典来存储每个图块的特征
        tile_features = {}
        
        for item in unique_elements:
            # 获取每个图块对应的精灵信息
            sprites = pyboy.get_sprite_by_tile_identifier([item])
            
            # 找出这个图块在矩阵中的所有位置
            positions = np.where(game_area == item)
            
            # 计算这个图块的特征
            feature = {
                'count': len(positions[0]),  # 出现次数
                'positions': list(zip(positions[1], positions[0])),  # 所有位置
                'sprites': sprites,  # 精灵信息
                'is_moving': False,  # 是否移动
                'is_unique': len(positions[0]) == 1  # 是否唯一
            }
            
            tile_features[item] = feature
            
            # 打印特征信息
            print(f"\n图块值 {item}:")
            print(f"  出现次数: {feature['count']}")
            print(f"  是否唯一: {feature['is_unique']}")
            print(f"  精灵信息: {sprites}")
            
            # 如果这个图块只出现一次，可能是玩家或特殊敌人
            if feature['is_unique']:
                print(f"  可能位置: {feature['positions'][0]}")
            
            # 如果这个图块出现多次且位置分散，可能是普通敌人
            if feature['count'] > 1 and feature['count'] < 10:
                print(f"  多个位置: {feature['positions']}")

        # 分析可能的玩家和敌人位置
        print("\n可能的玩家和敌人位置:")
        for tile_id, feature in tile_features.items():
            if feature['is_unique']:
                print(f"图块 {tile_id} 可能是玩家或特殊敌人，位置: {feature['positions'][0]}")
            elif 1 < feature['count'] < 10:
                print(f"图块 {tile_id} 可能是普通敌人，位置: {feature['positions']}")

        return game_area, tile_features

# 使用示例
rom_path = r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb"
state = r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb.state"

game_area, tile_features = analyze_positions(rom_path, state)
