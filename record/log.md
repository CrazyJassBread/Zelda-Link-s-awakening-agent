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

