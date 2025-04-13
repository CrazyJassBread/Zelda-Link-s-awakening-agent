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


**v10**
Sprites on screen:
自己
Sprite [2]: Position: (73, 78), Shape: (8, 16), Tiles: (Tile: 0, Tile: 1), On screen: True
Sprite [3]: Position: (81, 78), Shape: (8, 16), Tiles: (Tile: 2, Tile: 3), On screen: True

4， 5是拿剑
喷药粉是变化的，所以暂且禁止这个按键，拿盾牌不变化
Sprite [17]: Position: (71, 85),Tiles: (Tile: Tile: 36, Tile: Tile: 37), On screen: True
Sprite [18]: Position: (77, 82),Tiles: (Tile: Tile: 36, Tile: Tile: 37), On screen: True
Sprite [19]: Position: (83, 84),Tiles: (Tile: Tile: 36, Tile: Tile: 37), On screen: True

单个的是钥匙，但是也在变化

敌人一般来说按照排除法是2个一组，但是因为可能出现帧率的问题只显示出一个，这里要注意.
""
Sprite [5]: Position: (115, 48),Tiles: (Tile: Tile: 4, Tile: Tile: 5), On screen: True
Sprite [13]: Position: (100, 20),Tiles: (Tile: Tile: 62, Tile: Tile: 63), On screen: True
Sprite [15]: Position: (114, 54),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [16]: Position: (122, 54),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
visible_sprites_count : 4, enemy_count : 1
""

主要代码在这一行
"""
                    sprite_info = f"Sprite [{idx}]: Position: ({sprite.x}, {sprite.y}),Tiles: (Tile: {sprite.tiles[0]}, Tile: {sprite.tiles[1]}), On screen: {sprite.on_screen}"

"""

"""
Sprite [13]: Position: (105, 40),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [14]: Position: (113, 40),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [15]: Position: (58, 36),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [16]: Position: (66, 36),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
---------------------------------------------------------------------
Sprite [15]: Position: (105, 40),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [16]: Position: (113, 40),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [17]: Position: (58, 36),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [18]: Position: (66, 36),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
---------------------------------------------------------------------
Sprite [17]: Position: (105, 40),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [18]: Position: (113, 40),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [19]: Position: (59, 36),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [20]: Position: (67, 36),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
---------------------------------------------------------------------
Sprite [19]: Position: (105, 41),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [20]: Position: (113, 41),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [21]: Position: (59, 36),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [22]: Position: (67, 36),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
---------------------------------------------------------------------
Sprite [13]: Position: (106, 41),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [14]: Position: (114, 41),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [15]: Position: (59, 36),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [16]: Position: (67, 36),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
---------------------------------------------------------------------
Sprite [15]: Position: (106, 41),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [16]: Position: (114, 41),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [17]: Position: (60, 36),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [18]: Position: (68, 36),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
---------------------------------------------------------------------
Sprite [17]: Position: (106, 41),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [18]: Position: (114, 41),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [19]: Position: (60, 36),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [20]: Position: (68, 36),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
---------------------------------------------------------------------
Sprite [19]: Position: (107, 42),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [20]: Position: (115, 42),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [21]: Position: (60, 36),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [22]: Position: (68, 36),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
---------------------------------------------------------------------
Sprite [13]: Position: (107, 42),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [14]: Position: (115, 42),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [15]: Position: (61, 37),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [16]: Position: (69, 37),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
---------------------------------------------------------------------
Sprite [15]: Position: (107, 43),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [16]: Position: (115, 43),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [17]: Position: (61, 37),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [18]: Position: (69, 37),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
---------------------------------------------------------------------
Sprite [17]: Position: (108, 43),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [18]: Position: (116, 43),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [19]: Position: (62, 37),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [20]: Position: (70, 37),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
---------------------------------------------------------------------
Sprite [19]: Position: (108, 43),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [20]: Position: (116, 43),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [21]: Position: (62, 37),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [22]: Position: (70, 37),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
---------------------------------------------------------------------
Sprite [13]: Position: (108, 44),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [14]: Position: (116, 44),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [15]: Position: (62, 37),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [16]: Position: (70, 37),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
---------------------------------------------------------------------
Sprite [15]: Position: (109, 44),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [16]: Position: (117, 44),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [17]: Position: (63, 37),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [18]: Position: (71, 37),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
---------------------------------------------------------------------
Sprite [17]: Position: (109, 44),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [18]: Position: (117, 44),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [19]: Position: (63, 38),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [20]: Position: (71, 38),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
---------------------------------------------------------------------
Sprite [19]: Position: (109, 45),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [20]: Position: (117, 45),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [21]: Position: (64, 38),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
Sprite [22]: Position: (72, 38),Tiles: (Tile: Tile: 68, Tile: Tile: 69), On screen: True
---------------------------------------------------------------------

"""
不动-直接被敌人杀死的情况下，结果呈现规律性，

"""
Sprite [13]: Position: (13, 63),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [14]: Position: (21, 63),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
---------------------------------------------------------------------
Sprite [15]: Position: (9, 63),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [16]: Position: (17, 63),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
---------------------------------------------------------------------
Sprite [17]: Position: (5, 63),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [18]: Position: (13, 63),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
---------------------------------------------------------------------
Sprite [19]: Position: (1, 63),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [20]: Position: (9, 63),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
---------------------------------------------------------------------
Sprite [13]: Position: (-3, 63),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
Sprite [14]: Position: (5, 63),Tiles: (Tile: Tile: 70, Tile: Tile: 71), On screen: True
"""
杀死一个敌人，结果也呈现规律，
最终以此来判断敌人是否死亡。
"""
---------------------------------------------------------------------
Sprite [13]: Position: (84, 16),Tiles: (Tile: Tile: 36, Tile: Tile: 37), On screen: True
Sprite [15]: Position: (50, 19),Tiles: (Tile: Tile: 30, Tile: Tile: 31), On screen: True
Sprite [16]: Position: (58, 19),Tiles: (Tile: Tile: 30, Tile: Tile: 31), On screen: True
---------------------------------------------------------------------
Sprite [15]: Position: (84, 16),Tiles: (Tile: Tile: 36, Tile: Tile: 37), On screen: True
Sprite [17]: Position: (50, 19),Tiles: (Tile: Tile: 30, Tile: Tile: 31), On screen: True
Sprite [18]: Position: (58, 19),Tiles: (Tile: Tile: 30, Tile: Tile: 31), On screen: True
---------------------------------------------------------------------
Sprite [17]: Position: (84, 16),Tiles: (Tile: Tile: 36, Tile: Tile: 37), On screen: True
Sprite [19]: Position: (50, 19),Tiles: (Tile: Tile: 30, Tile: Tile: 31), On screen: True
Sprite [20]: Position: (58, 19),Tiles: (Tile: Tile: 30, Tile: Tile: 31), On screen: True
---------------------------------------------------------------------
Sprite [19]: Position: (84, 16),Tiles: (Tile: Tile: 36, Tile: Tile: 37), On screen: True
Sprite [21]: Position: (49, 18),Tiles: (Tile: Tile: 30, Tile: Tile: 31), On screen: True
Sprite [22]: Position: (57, 18),Tiles: (Tile: Tile: 30, Tile: Tile: 31), On screen: True
"""

"""
Sprite [13]: Position: (53, 21),Tiles: (Tile: Tile: 62, Tile: Tile: 63), On screen: True
---------------------------------------------------------------------
Sprite [15]: Position: (52, 20),Tiles: (Tile: Tile: 62, Tile: Tile: 63), On screen: True
---------------------------------------------------------------------
Sprite [17]: Position: (53, 21),Tiles: (Tile: Tile: 62, Tile: Tile: 63), On screen: True
---------------------------------------------------------------------
Sprite [19]: Position: (52, 20),Tiles: (Tile: Tile: 62, Tile: Tile: 63), On screen: True
---------------------------------------------------------------------
Sprite [13]: Position: (53, 21),Tiles: (Tile: Tile: 62, Tile: Tile: 63), On screen: True
---------------------------------------------------------------------
Sprite [15]: Position: (52, 20),Tiles: (Tile: Tile: 62, Tile: Tile: 63), On screen: True
"""

出现三个,一个，合理怀疑是不是敌人得血量减少一半就只出现一半（也就是一个sprite）,还是不太确定，出剑得4, 5有的时候还是出现一个5,所以很难说
但是总体上还是根据整除公式直接计算了.

结果不知道是位置没设置好，总是会向下调到黑洞里面，钥匙的位置还要再改一版

钥匙的具体位置：
""
---------------------------------------------------------------------
Frame 7607: position : 13 :( 34,  43 ) , 14 : ( 42, 43 )
------------------------------------------------
""

因此钥匙坐标取平均（38， 43）

| 奖励类型 | 条件 | 数值 | 说明 |
|---------|------|------|------|
| 位置奖励 | 靠近钥匙(距离<10) | +300 | 非常接近钥匙时给予高额奖励 |
| | 靠近钥匙(距离<20) | +200 | 较为接近钥匙时给予中等奖励 |
| | 靠近钥匙(距离<30) | +100 | 开始接近钥匙时给予基础奖励 |
| | 距离改进 | 距离减少×50 | 每次向钥匙方向移动给予正比于移动距离的奖励 |
| 区域奖励 | 在指定区域内(2.1/2.2/2.3) | +10 | 在游戏地图合理区域内活动给予小额奖励 |
| | 在指定区域外 | -50 | 离开合理区域给予惩罚 |
| 探索奖励 | 访问新位置 | +30 | 鼓励探索未访问过的位置 |
| 房间位置 | 在正确房间(0x3a) | +200 | 在含有钥匙的房间给予奖励 |
| | 不在正确房间 | -100 | 不在含有钥匙的房间给予惩罚 |
| 生命值 | 生命值变化 | 变化值×100 | 生命值增加获得奖励，减少获得惩罚 |
| 钥匙奖励 | 获得钥匙 | +1000 | 获得钥匙时给予最高奖励，并结束回合 |
| 敌人奖励 | 杀死一个敌人 | +200 | 消灭一个敌人时给予奖励 |
| | 杀死两个敌人 | +500 | 消灭两个敌人时给予高额奖励 |
| | 多于一个敌人存在 | -10 | 有多个敌人时给予轻微惩罚，鼓励战斗 |

最大：的效应。