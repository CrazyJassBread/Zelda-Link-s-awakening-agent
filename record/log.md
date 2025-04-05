位置相关信息：

1. 钥匙位置 ： （37， 45）
2. 避免陷阱的位置，这里写的是安全距离
    2.1 x <= 119 && x >= 24, y <= 36 &&  y >= 22
    2.2 0 <= x <= 113 , 36 <= y <= 65
    2.3 24 <= x <= 119, 65 <= y <= 81 

-- 环境就直接是 3_12, 不是zelda环境


不知道为什么，好像加了边界不掉到坑里的内容就故意回去边界， 不知道为什么，好奇怪，没有预料到的时间减少，甚至不收敛，可是之前那个版本都收敛了。


**v6**
```
Traceback (most recent call last):
  File "d:\codes\codes_pycharm\da_chuang\codes\zelda_env_v6.py", line 392, in <module>
    action, _states = model.predict(obs)
                      ^^^^^^^^^^^^^^^^^^
  File "D:\Tool\python_3_12\Lib\site-packages\stable_baselines3\common\base_class.py", line 557, in predict     
    return self.policy.predict(observation, state, episode_start, deterministic)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Tool\python_3_12\Lib\site-packages\stable_baselines3\common\policies.py", line 357, in predict       
    raise ValueError(
ValueError: You have passed a tuple to the predict() function instead of a Numpy array or a Dict. You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) vs `obs = vec_env.reset()` (SB3 VecEnv). See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
```
不知道，但是v7跑的不错
注意奖励函数最好不要设置成全局的那种。
还是有问题，跑到最后会出现跑的速度过快然后没来得及与敌人交手就跑过去然后被敌人推下去的情况

**v8** :
下面是我对代码中奖励函数和step方法的分析，展示了各操作与奖励之间的关系：
| 操作类型 | 具体行为 | 关联奖励 | 奖励值 | 解释 |
|---------|---------|---------|-------|------|
| 位置奖励 | 接近钥匙 | position_reward | 距离改善 × 50 | 当智能体向钥匙位置移动时，根据距离改善程度给予正奖励 |
| 阶段性位置奖励 | 接近钥匙达到阈值 | position_reward | +300/+200/+100 | 当距离钥匙<10/20/30单位时给予额外奖励，鼓励更接近目标 |
| 区域奖励 | 在指定区域内 | area_reward | +10 | 当位于游戏中特定区域(2.1/2.2/2.3)内获得奖励 |
| 区域惩罚 | 离开指定区域 | area_reward | -50 | 当离开目标区域时给予惩罚，防止迷路 |
| 探索奖励 | 访问新位置 | exploration_reward | +30 | 当访问以前未到过的位置时获得奖励，促进地图探索 |
| 房间奖励 | 在正确房间内 | room_reward | +200/-100 | 在房间0x3a中+200分，其他房间-100分 |
| 钥匙奖励 | 获得钥匙 | key_reward | +1000 | 成功获取钥匙时的主要目标奖励 |
| 生命值奖励 | 生命值变化 | health_reward | 变化值 × 100 | 生命值增加获得正奖励，减少则获得负奖励 |
| 按键惩罚 | 生命稳定时使用A/B键 | button_penalty | -50 | 当生命值稳定时使用A/B键给予惩罚 |
| 阶段性按键奖励 | 战斗阶段使用A/B键 | 阶段奖励 | +20 | 在战斗阶段使用A/B键获得额外奖励 |
| 阶段性按键惩罚 | 探索阶段使用A/B键 | 阶段惩罚 | -50 | 在探索阶段使用A/B键给予额外惩罚 |

代码基本上没什么问题。