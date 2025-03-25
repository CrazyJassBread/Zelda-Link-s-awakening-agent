from pyboy import PyBoy
import numpy as np
from colorama import init, Fore, Style
from PIL import Image

# Initialize colorama
init()

# Define the file paths for the ROM and the game state
rom_path = r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb"
init_state_path = r"D:\codes\codes_pycharm\da_chuang\Legend_of_Zelda\Legend_of_Zelda.gb.state"

# Initialize PyBoy simulator
pyboy = PyBoy(rom_path, window="SDL2")

# Try loading the initial game state
try:
    with open(init_state_path, "rb") as f:
        pyboy.load_state(f)
        print(f"Successfully loaded game state from {init_state_path}")
except FileNotFoundError:
    print(f"Warning: Game state file {init_state_path} not found, starting a new game.")

# Initialize the previous game area matrix for comparison
prev_game_area_matrix = pyboy.game_area()

# Set NumPy to display the entire matrix
np.set_printoptions(threshold=np.inf)

# Open a file to store the output
with open('output.txt', 'a') as f:
    # Main loop to monitor game area matrix changes
    while pyboy.tick():
        # Get the current game area matrix
        current_game_area_matrix = pyboy.game_area()

        # Compare the current and previous game area matrices to find the changes
        changed_positions = np.argwhere(current_game_area_matrix != prev_game_area_matrix)

        # Format and print the current game area matrix, with changes highlighted in red
        formatted_matrix = []
        for i in range(current_game_area_matrix.shape[0]):
            row = []
            for j in range(current_game_area_matrix.shape[1]):
                # Highlight only the changed values in red
                if (i, j) in changed_positions:
                    element_str = f"{Fore.RED}{current_game_area_matrix[i, j]}{Style.RESET_ALL}"
                else:
                    element_str = str(current_game_area_matrix[i, j])
                row.append(element_str)
            formatted_matrix.append(" ".join(row))

        # Combine the formatted matrix into a single string for output
        matrix_str = "\n".join(formatted_matrix)

        # Print and record the current game area matrix
        result_str = f"当前游戏区域矩阵（帧号: {pyboy.frame_count}）:\n{matrix_str}\n"
        print(result_str)
        f.write(result_str.replace(Fore.RED, "").replace(Style.RESET_ALL, ""))  # Remove color codes for text file

        # If there are changes, print and record the positions and values of the changes
        if changed_positions.size > 0:
            matrix_change_info = "游戏区域矩阵发生变化的位置及值:\n"
            print(matrix_change_info)
            f.write(matrix_change_info)
            for pos in changed_positions:
                row, col = pos
                old_value = prev_game_area_matrix[row, col]
                new_value = current_game_area_matrix[row, col]
                position_change_info = f"位置: ({row}, {col}), 旧值: {old_value}, 新值: {new_value}\n"
                print(position_change_info)
                f.write(position_change_info)

            # Capture the current screen image and save it as a PNG file
            screenshot_path = f"game_screenshot_frame_{pyboy.frame_count}.png"
            screen_image = pyboy.screen.image # Get the current screen
            image = Image.fromarray(screen_image)  # Convert it to a PIL Image
            image.save(screenshot_path)  # Save the image as a PNG
            print(f"Screenshot saved as {screenshot_path}")

        # Update the previous game area matrix
        prev_game_area_matrix = current_game_area_matrix

# Stop the simulator
pyboy.stop()
