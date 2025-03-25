from pyboy import PyBoy
pyboy = PyBoy('D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb')
while pyboy.tick():
    pass
pyboy.stop()