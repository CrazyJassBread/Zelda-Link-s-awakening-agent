"""
测试训练好的Zelda环境PPO模型
"""

import gym
from gym import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from pathlib import Path
import time
# 导入多种算法
from stable_baselines3 import PPO, A2C, SAC, DQN
import matplotlib.pyplot as plt
# 如果要使用Rainbow DQN，需要安装sb3-contrib
try:
    from sb3_contrib import QRDQN
except ImportError:
    print("要使用高级DQN变种，请安装sb3-contrib: pip install sb3-contrib")

# 导入ZeldaEnv类
from zelda_env_v12 import ZeldaEnv

def main():
    # 配置参数
    config = {
        "session_path": r"D:\codes\codes_pycharm\da_chuang\sessions",  # 存储路径
        "headless": False,  # 测试时使用有界面模式，可以观察AI行为
        "action_freq": 1,  # 动作频率
        "max_steps": 200000,  # 最大步骤数
        "init_state": r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb.state",  # 初始化状态文件路径
        "algorithm": "PPO",  # 可选: "PPO", "A2C", "SAC", "DQN", "QRDQN"
    }

    # 创建测试环境
    env = ZeldaEnv(rom_path=r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb", config=config)
    
    # 设置模型加载路径
    algorithm = config["algorithm"]
    model_save_dir = Path(config["session_path"]) / "zelda_models"
    
    # 获取最新的训练文件夹
    train_folders = sorted([d for d in model_save_dir.iterdir() if d.is_dir() and d.name.startswith(f"{algorithm}_")])
    if not train_folders:
        raise FileNotFoundError(f"未找到任何{algorithm}训练文件夹")
    
    latest_train_folder = train_folders[-1]
    print(f"使用最新的训练文件夹: {latest_train_folder}")

    # 加载最终模型
    model_path = latest_train_folder / "final_model.zip"
    if not model_path.exists():
        # 尝试加载最新的checkpoint
        checkpoints = sorted(list((latest_train_folder / "checkpoints").glob(f"{algorithm}_model_*.zip")), 
                           key=lambda x: int(x.stem.split('_')[-1]))
        if checkpoints:
            model_path = checkpoints[-1]
        else:
            raise FileNotFoundError("未找到任何可用的模型文件")
    
    print(f"加载模型: {model_path}")
    
    # 根据算法加载对应的模型
    if algorithm == "PPO":
        model = PPO.load(model_path, env=env)
    elif algorithm == "A2C":
        model = A2C.load(model_path, env=env)
    elif algorithm == "DQN":
        model = DQN.load(model_path, env=env)
    elif algorithm == "QRDQN" and 'QRDQN' in globals():
        model = QRDQN.load(model_path, env=env)
    else:
        print(f"不支持的算法: {algorithm}，使用默认的PPO")
        model = PPO.load(model_path, env=env)
    
    # 记录测试结果
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    total_episodes = 10  # 测试10个episode
    
    # 添加动作统计
    action_counts = {i: 0 for i in range(env.action_space.n)}
    
    for episode in range(total_episodes):
        print(f"开始测试 Episode {episode+1}/{total_episodes}")
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=False)
            action_counts[action] += 1  # 记录动作使用频率
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # 可选：减慢测试速度，便于观察
            time.sleep(0.01)
            
            if done:
                success_count += 1
                print(f"Episode {episode+1} 成功! 奖励: {episode_reward:.2f}, 步数: {episode_length}")
            
            if episode_length >= 5000:  # 防止测试时间过长
                print(f"Episode {episode+1} 达到最大步数限制")
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    # 打印测试结果统计
    print("\n测试结果统计:")
    print(f"成功率: {success_count}/{total_episodes} ({success_count/total_episodes*100:.2f}%)")
    print(f"平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"平均步数: {np.mean(episode_lengths):.2f}")
    
    # 打印动作使用统计
    print("\n动作使用统计:")
    for action, count in action_counts.items():
        try:
            action_name = env.valid_actions[action].__str__().split('.')[-1]  # 获取动作名称
        except:
            action_name = f"Action_{action}"
        percentage = (count / sum(action_counts.values())) * 100
        print(f"动作 {action} ({action_name}): {count}次 ({percentage:.1f}%)")
    
    # 绘制测试奖励图表
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, total_episodes+1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{algorithm} Test Episodes Rewards')
    plt.savefig(str(latest_train_folder / "test_rewards.png"))
    plt.show()
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()