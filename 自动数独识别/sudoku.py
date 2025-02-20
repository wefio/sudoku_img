import numpy as np
import cv2
import pytesseract as pyt
import matplotlib.pyplot as plt
from collections import Counter
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 导入 tqdm


# ============================ 用户可修改部分 ============================
# 设置 Tesseract 可执行文件路径，确保路径正确
pyt.pytesseract.tesseract_cmd = r'C:/Application/Tesseract-OCR/tesseract.exe'  # 修改为 Tesseract 安装路径

# 图像文件路径
image_path = 'sudoku5.png'  # 修改为你的数独图像路径

# 数独格子大小，默认为 9x9
grid_size = (9, 9)  # 根据需求修改，如果是其他大小的数独，可以修改这里

# OCR识别的配置参数，可以调整以优化识别效果
ocr_config = '-c tessedit_char_whitelist=123456789 --oem 3 --psm 6 outputbase digits'  # 可根据需要修改 OCR 参数


# 切割图像时裁剪比例，0.1 表示每个格子的 10% 边缘被裁剪掉
crop_ratio = 0.05  # 修改裁剪比例来调整裁剪的边缘大小

# =======================================================================

# 图像预处理函数
def adaptive_binarize(image):
    """自适应直方图二值化处理"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

def denoise_image(binary_image):
    """去噪处理"""
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

def find_sudoku_outer_frame(image):
    """通过轮廓检测找到数独最外框"""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return approx
    return None

def perspective_transform(image, pts):
    width, height = 400, 400
    pts2 = np.array([[0, 0], [0, width], [width, height], [height, 0]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, pts2)
    return cv2.warpPerspective(image, matrix, (width, height))

def crop_image(image):
    """裁剪去除图像中的无关部分"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return image[y:y+h, x:x+w]


def crop_cell(cell_image, crop_ratio=0.1):
    """裁剪格子边缘，只保留数字区域"""
    height, width = cell_image.shape[:2]
    crop_top, crop_bottom = int(height * crop_ratio), int(height * (1 - crop_ratio))
    crop_left, crop_right = int(width * crop_ratio), int(width * (1 - crop_ratio))
    return cell_image[crop_top:crop_bottom, crop_left:crop_right]

def enhance_image_for_ocr(cell_image):
    """增强图像，准备 OCR 识别"""
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return cv2.medianBlur(binary, 3)

def recognize_number_from_image(cell_image):
    """使用 Tesseract 识别单个小格子中的数字"""
    enhanced_image = enhance_image_for_ocr(cell_image)
    return pyt.image_to_string(enhanced_image, config=ocr_config).strip()

def split_image_into_grid(image, grid_size=(9, 9), vertical_offset=3, horizontal_offset=3):  # 修改偏移量
    """切分图像为 9x9 格子并应用偏移"""
    height, width = image.shape[:2]
    cell_height, cell_width = height // grid_size[0], width // grid_size[1]
    grid_cells = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x1, y1 = j * cell_width + horizontal_offset, i * cell_height + vertical_offset
            x2, y2 = (j + 1) * cell_width + horizontal_offset, (i + 1) * cell_height + vertical_offset
            grid_cells.append((i, j, image[y1:y2, x1:x2]))
    return grid_cells

# 使用多线程识别每个格子
def recognize_number_from_image_threaded(cell_image):
    """多线程处理，识别一个格子中的数字"""
    return recognize_number_from_image(cell_image)

def display_grid_cells_with_numbers(grid_cells, grid_size=(9, 9), crop_ratio=0.15, num_recognitions=2):
    """标记识别的数字并返回数独矩阵"""
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
    axes = axes.flatten() # type: ignore
    sudoku_grid = [['.' for _ in range(grid_size[1])] for _ in range(grid_size[0])]

    # 使用 ThreadPoolExecutor 创建线程池
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_cell = {executor.submit(recognize_number_from_image, crop_cell(cell, crop_ratio)): (i, j, cell) 
                          for i, j, cell in grid_cells}

        # 使用 tqdm 创建进度条
        for future in tqdm(as_completed(future_to_cell), total=len(future_to_cell), desc="识别进度", unit="格"):
            i, j, cell = future_to_cell[future]
            try:
                digit = future.result()

                # 使用灰度图像，并在上面绘制数字
                gray_cell = cv2.cvtColor(crop_cell(cell, crop_ratio), cv2.COLOR_BGR2GRAY)  # 转为灰度图
                if digit.isdigit():
                    sudoku_grid[i][j] = digit
                    # 在灰度图上绘制数字（红色）
                    cv2.putText(gray_cell, digit, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                # 显示当前格子
                ax = axes[i * grid_size[1] + j]
                ax.imshow(gray_cell, cmap='gray')  #显示
                ax.set_title(f"Cell ({i}, {j})")
                ax.axis('off')
                
            except Exception as e:
                print(f"Error processing cell ({i}, {j}): {e}")

    plt.tight_layout()
    plt.show()
    return sudoku_grid

# 数独求解算法
def is_valid(board, row, col, num):
    """判断填入的数字是否合法"""
    for x in range(9):
        if board[row][x] == num or board[x][col] == num:
            return False
    start_row, start_col = row - row % 3, col - col % 3
    for i in range(3):
        for j in range(3):
            if board[i + start_row][j + start_col] == num:
                return False
    return True

def solve_sudoku(board):
    """回溯法解数独"""
    empty_cell = find_empty_location(board)
    if not empty_cell:
        return True  # 已解完
    row, col = empty_cell
    for num in map(str, range(1, 10)):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = '.'  # 回溯
    return False

def find_empty_location(board):
    """查找空白位置"""
    for i in range(9):
        for j in range(9):
            if board[i][j] == '.':
                return i, j
    return None

# ========================== 主程序执行 ============================

if __name__ == "__main__":
    # 读取并预处理图像
    image = cv2.imread(image_path)
    processed_image = adaptive_binarize(image)
    denoised_image = denoise_image(processed_image)

    cv2.imshow('origin', image)
    cv2.imshow('processed', processed_image)
    cv2.imshow('denoised', denoised_image)

    # 查找数独外框
    outer_frame = find_sudoku_outer_frame(denoised_image)
    if outer_frame is None:
        print("未找到数独外框！")
    else:
        transformed_image = perspective_transform(image, np.float32(outer_frame.reshape(4, 2)))
        cropped_image = crop_image(transformed_image)
        cv2.imshow('Cropped Image', cropped_image)

        # 切割为格子并识别数字
        grid_cells = split_image_into_grid(cropped_image, grid_size=grid_size)
        sudoku_grid = display_grid_cells_with_numbers(grid_cells, grid_size=grid_size, crop_ratio=crop_ratio)

        # 打印识别结果
        print("\n识别结果的数独：")
        for row in sudoku_grid:
            print(" ".join(row))
        
        # 解数独
        print("\n尝试求解数独...")
        if solve_sudoku(sudoku_grid):
            print("解出的数独：")
            for row in sudoku_grid:
                print(" ".join(row))
        else:
            print("无法解出数独！")