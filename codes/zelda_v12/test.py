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
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# 导入ZeldaEnv类
from zelda_env_v12 import ZeldaEnv

# 修改模型加载逻辑
def main():
    # 配置参数
    config = {
        "session_path": r"D:\codes\codes_pycharm\da_chuang\sessions",  # 存储路径
        "headless": False,  # 测试时使用有界面模式，可以观察AI行为
        "action_freq": 1,  # 动作频率
        "max_steps": 200000,  # 最大步骤数
        "init_state": r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb.state"  # 初始化状态文件路径
    }

    # 创建测试环境
    env = ZeldaEnv(rom_path=r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb", config=config)
    
    # 设置模型加载路径
    model_save_dir = Path(config["session_path"]) / "zelda_models"
    
    # 获取最新的训练文件夹
    train_folders = sorted([d for d in model_save_dir.iterdir() if d.is_dir() and d.name.startswith("train_")])
    if not train_folders:
        raise FileNotFoundError("未找到任何训练文件夹")
    
    latest_train_folder = train_folders[-1]
    print(f"使用最新的训练文件夹: {latest_train_folder}")

    # 优先加载final_model
    model_path = latest_train_folder / "final_model.zip"
    if model_path.exists():
        print("使用最终模型")
    else:
        # 按步数排序checkpoint
        checkpoints = sorted(latest_train_folder.glob("zelda_model_*.zip"), 
                           key=lambda x: int(x.stem.split('_')[-1]))
        if not checkpoints:
            raise FileNotFoundError("未找到任何可用的模型文件")
        # 使用倒数第二个checkpoint（最后一个可能不完整）
        model_path = checkpoints[-2] if len(checkpoints) > 1 else checkpoints[-1]
        print(f"使用checkpoint模型: {model_path}")

    print(f"加载模型: {model_path}")
    model = PPO.load(model_path, env=env)
    
    # 记录测试结果
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    total_episodes = 10  # 测试10个episode
    
    # 在主循环开始前打印环境信息
    print("\n环境和模型信息:")
    print(f"动作空间: {env.action_space}")
    print(f"观察空间: {env.observation_space}")
    print(f"模型架构: {model.policy}")
    for episode in range(total_episodes):
        print(f"开始测试 Episode {episode+1}/{total_episodes}")
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)

             # 打印动作信息帮助调试
            print(f"Step {episode_length}: Action={action}, Reward={reward:.2f}")
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
    
    # 绘制测试奖励图表
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, total_episodes+1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Test Episodes Rewards')
    plt.savefig(f"{model_save_dir}/test_rewards.png")
    plt.show()
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()