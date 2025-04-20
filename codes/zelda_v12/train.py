"""
训练Zelda环境的PPO模型，并使用EvalCallback定期评估模型性能
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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime

from zelda_env_v12 import ZeldaEnv


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

        self.model_save_dir = None  # 添加这行

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            if len(self.episode_rewards) % 1000 == 0 and self.model_save_dir is not None:
                print(f"保存奖励曲线图到：{self.model_save_dir}/rewards_curve_{len(self.episode_rewards)}.png")
                self.plot_rewards()

        return True
    
    def plot_rewards(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.episode_rewards, label='Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards over Episodes')
        plt.legend()
        plt.savefig(f"{self.model_save_dir}/rewards_curve_{len(self.episode_rewards)}.png")
        plt.close()

def main():
    config = {
        "session_path": r"D:\codes\codes_pycharm\da_chuang\sessions",  # 存储路径
        "headless": False,  # 训练时使用无界面模式
        "action_freq": 1,  # 动作频率
        "max_steps": 200000,  # 最大步骤数
        "init_state": r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb.state",  # 初始化状态文件路径
        "use_eval" : False,
    }

    env = ZeldaEnv(rom_path=r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb", config=config)
    # env = Monitor(env, config["session_path"])

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_dir = Path(config["session_path"]) / "zelda_models" / f"train_{current_time}"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1,
        learning_rate=5e-5,        # 进一步降低学习率
        n_steps=4096,              # 增加步数以获取更多经验
        batch_size=256,            # 增加批量大小
        gamma=0.99,                
        ent_coef=0.005,           # 减小探索系数
        clip_range=0.1,           # 减小裁剪范围提高稳定性
    )
    
    # 创建回调函数
    callbacks = []

    # 1. 定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(model_save_dir),
        name_prefix="zelda_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

        # 2. 定期评估模型性能
    # 2. 定期评估模型性能（仅在电脑性能好的情况下使用）
    if config["use_eval"]:
        print("启用评估环境进行定期评估...")
        eval_config = config.copy()
        eval_config["headless"] = True  # 评估时使用无界面模式以减少资源占用
        eval_env = ZeldaEnv(rom_path=r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb", config=eval_config)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_save_dir / "best_model"),
            log_path=str(model_save_dir / "eval_logs"),
            eval_freq=10000,  # 每10000步评估一次
            deterministic=True,
            render=False,
            n_eval_episodes=5  # 每次评估5个episode
        )
        callbacks.append(eval_callback)
    else:
        print("跳过评估环境，以减少资源占用...")
        eval_env = None

    # 3. 记录奖励
    reward_callback = RewardCallback()
    reward_callback.model_save_dir = model_save_dir

    callbacks.append(reward_callback)
    
    # 训练模型，使用多个回调函数
    model.learn(
        total_timesteps=1000000, 
        callback=callbacks,
        tb_log_name="PPO_zelda"
    )

     # 训练完成后保存最终模型
    final_model_path = model_save_dir / "final_model.zip"
    model.save(final_model_path)
    print(f"最终模型已保存至: {final_model_path}")
    
    # 绘制完整的奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(reward_callback.episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Complete Training Reward Curve')
    plt.savefig(str(model_save_dir / "final_reward_curve.png"))
    plt.show()

    # 关闭环境
    env.close()
    
    if eval_env is not None:
        eval_env.close()    
if __name__ == "__main__":
    main()  
    
    
    

"""
def main():
    # ... 现有代码 ...
    
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1,
        learning_rate=1e-5,        # 进一步降低学习率
        n_steps=8192,              # 增加步数以获取更多经验
        batch_size=512,            # 增加批量大小
        gamma=0.995,               # 增加折扣因子
        ent_coef=0.001,           # 进一步减小探索
        clip_range=0.1,           
        n_epochs=20               # 增加每批数据的训练轮数
    )

    # ... 其他代码 ...

    # 增加训练步数
    model.learn(
        total_timesteps=2000000,  # 增加到200万步
        callback=callbacks
    )
"""