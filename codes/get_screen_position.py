import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import time
import matplotlib.pyplot as plt

class PlayerPositionChecker:
    def __init__(self, rom_path, state_path=None):
        # 初始化PyBoy
        self.pyboy = PyBoy(rom_path, window="SDL2")
        
        # 如果有存档，加载存档
        if state_path:
            with open(state_path, "rb") as f:
                self.pyboy.load_state(f)
        
        # 设置模拟器速度
        self.pyboy.set_emulation_speed(1)
        
        # 初始化动作映射
        self.actions = {
            'up': WindowEvent.PRESS_ARROW_UP,
            'down': WindowEvent.PRESS_ARROW_DOWN,
            'left': WindowEvent.PRESS_ARROW_LEFT,
            'right': WindowEvent.PRESS_ARROW_RIGHT,
            'a': WindowEvent.PRESS_BUTTON_A,
            'b': WindowEvent.PRESS_BUTTON_B
        }
        
        # 用于记录位置历史
        self.position_history = []

    def get_player_position(self):
        """获取角色在屏幕上的位置"""
        # 获取游戏画面
        game_screen = np.array(self.pyboy.screen_image())
        
        # 显示当前帧的图像（用于调试）
        plt.imshow(game_screen)
        plt.title("Current Frame")
        plt.show()
        
        # 打印游戏画面的形状和一些像素值（用于调试）
        print("Screen shape:", game_screen.shape)
        print("Sample pixel values:")
        print(game_screen[80:100, 80:100])  # 打印中心区域的像素值
        
        # 获取角色特征（需要根据实际游戏画面调整）
        # 这里假设角色是白色的，您需要根据实际情况修改颜色值
        player_color = [255, 255, 255]
        
        # 找到匹配颜色的位置
        matches = np.where(np.all(game_screen == player_color, axis=2))
        
        if len(matches[0]) > 0:
            # 取中心点作为角色位置
            y = int(np.mean(matches[0]))
            x = int(np.mean(matches[1]))
            print(f"Found player at position: ({x}, {y})")
            return x, y
        
        print("Player not found!")
        return None, None

    def track_movement(self, duration=10):
        """追踪角色移动一段时间"""
        start_time = time.time()
        while time.time() - start_time < duration:
            # 获取位置
            pos = self.get_player_position()
            if pos[0] is not None:
                self.position_history.append(pos)
                print(f"Position: {pos}")
            
            # 等待一小段时间
            time.sleep(0.1)
            # 推进游戏帧
            self.pyboy.tick()

    def test_controls(self):
        """测试控制和位置检测"""
        print("测试控制和位置检测")
        print("使用方向键移动角色，按 'q' 退出")
        
        while True:
            # 获取并显示当前位置
            pos = self.get_player_position()
            print(f"Current position: {pos}")
            
            # 等待用户输入
            key = input("输入方向 (up/down/left/right/q): ")
            if key == 'q':
                break
            
            # 执行动作
            if key in self.actions:
                # 按下按键
                self.pyboy.send_input(self.actions[key])
                self.pyboy.tick()
                # 释放按键
                self.pyboy.send_input(self.actions[key] + 1)  # +1 获取对应的释放事件
                self.pyboy.tick()

    def plot_movement_history(self):
        """绘制移动轨迹"""
        if not self.position_history:
            print("No movement history recorded")
            return
            
        positions = np.array(self.position_history)
        plt.figure(figsize=(10, 10))
        plt.plot(positions[:, 0], positions[:, 1], 'b-')
        plt.scatter(positions[:, 0], positions[:, 1], c='r', s=50)
        plt.title("Player Movement History")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.show()

    def close(self):
        """关闭模拟器"""
        self.pyboy.stop()

def main():
    # 设置路径
    rom_path = r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb"
    state_path = r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb.state"
    
    # 创建检查器实例
    checker = PlayerPositionChecker(rom_path, state_path)
    
    try:
        # 测试模式选择
        print("选择测试模式：")
        print("1. 手动控制测试")
        print("2. 自动追踪测试")
        choice = input("请选择（1/2）：")
        
        if choice == '1':
            # 手动控制测试
            checker.test_controls()
        else:
            # 自动追踪测试
            print("开始追踪角色移动（10秒）...")
            checker.track_movement(10)
            checker.plot_movement_history()
        
    finally:
        # 确保正确关闭
        checker.close()

if __name__ == "__main__":
    main()
