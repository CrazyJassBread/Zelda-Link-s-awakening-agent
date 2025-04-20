
## 环境配置
1. 安装所需依赖：
看一下requirements.txt文件就行，好像只要特地安一个

```bash
pip install -r requirements.txt
```

2. 项目结构

```plaintext
d:\codes\codes_pycharm\da_chuang
├── codes
│   └── zelda_v12
│       ├── train.py      # PPO算法训练脚本
│       ├── train_mul.py  # 多算法训练脚本, 在这里更换算法
│       ├── test.py       # PPO算法测试脚本
│       ├── test_mul.py   # 多算法测试脚本， 同样在这里更换算法
│       └── zelda_env_v12.py  # 环境定义
├── Legend_of_Zelda
│   ├── Legend_of_Zelda.gb      # 游戏ROM
│   └── Legend_of_Zelda.gb.state # 初始状态文件
├── sessions
│   └── zelda_models     # 模型保存目录, 这里是用算法名称和时间来创建文件夹的，注意！
└── requirements.txt
```

3. 训练模型 (train_mul.py)
具体在
```python
config = {
    # ... 其他配置 ...
    "algorithm": "PPO",  # 可选: "PPO", "A2C", "SAC", "DQN", "QRDQN"
}
```

4. 训练结果保存

- 所有训练结果保存在 sessions/zelda_models 目录下
- 每次训练会创建一个新的文件夹，格式为： 算法名_时间戳 （例如： PPO_20240101_120000 ）
- 文件夹包含：
- checkpoints/ : 训练过程中的检查点模型
- final_model.zip : 训练完成的最终模型
- rewards_curve_*.png : 训练过程的奖励曲线图
- final_reward_curve.png : 完整训练过程的奖励曲线


测试同样
