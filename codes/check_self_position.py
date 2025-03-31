from pyboy import PyBoy
from pyboy.utils import WindowEvent
import time

def manual_memory_scan(pyboy, target_value, start=0xA000, end=0xDFFF):
    """手动扫描内存区域"""
    found = []
    for addr in range(start, end+1):
        try:
            if pyboy.memory[addr] == target_value:
               found.append(addr)
        except:
            continue
    return found

# 初始化并扫描
state = r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb.state"
pyboy = PyBoy(r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb",window = "SDL2")
#pyboy.game_wrapper.start_game()
try:
    with open(state, "rb") as f:
        pyboy.load_state(f)
except FileNotFoundError:
     print("No existing save file, starting new game")

# 查找值为93的地址
# 80， 88
target = 58
addresses = manual_memory_scan(pyboy, target)

print(f"找到 {len(addresses)} 个地址存储值 {target}:")
for addr in addresses:
    print(f"0x{addr:04X} -> {pyboy.memory[addr]}")

# # 实时监控示例（选一个地址）
# if len(addresses) > 0:
#     monitor_addr = addresses[0]
#     for i in range(100000000): # 监控60帧
#         if i % 10 == 0:
#             val = pyboy.memory[monitor_addr]
#             print(f"地址 0x{monitor_addr:04X} 当前值: {val}")
#         pyboy.tick()
#         time.sleep(1 /60)

# pyboy.stop()

# 实时监控所有找到的地址
if len(addresses) > 0:
    print(f"\n开始监控 {len(addresses)} 个地址:")
    for i in range(100000000):  # 监控60帧
        if i % 10 == 0:
            print("\n当前所有地址的值:")
            for addr in addresses:
                val = pyboy.memory[addr]
                print(f"地址 0x{addr:04X} 当前值: {val}")
            print("-" * 30)  # 添加分隔线使输出更清晰
        pyboy.tick()
        time.sleep(1/60)
else:
    print("没有找到符合条件的地址")

pyboy.stop()