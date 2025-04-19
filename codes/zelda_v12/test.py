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
    
    # 加载最佳模型（如果存在）
    best_model_path = model_save_dir / "best_model" / "best_model.zip"
    if best_model_path.exists():
        model_path = best_model_path
        print(f"加载最佳模型: {model_path}")
    else:
        # 否则加载最终模型
        model_path = model_save_dir / "zelda_final_model.zip"
        print(f"加载最终模型: {model_path}")
    
    # 加载训练好的模型
    model = PPO.load(model_path, env=env)
    
    # 记录测试结果
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    total_episodes = 10  # 测试10个episode
    
    for episode in range(total_episodes):
        print(f"开始测试 Episode {episode+1}/{total_episodes}")
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
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