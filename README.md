# 实验一 自动提取数独区域并进行透视变换
通过图像处理技术，从一张包含数独谜题的图像中自动检测并提取出数独的区域，并通过透视变换得到该区域的正面视图。
# 编程软件：Visual Studio Code（简称 vscode）
教程可参考 [VScode中配置Python运行环境](bilibili.com/video/BV1tF411M7hy)
# 实验步骤
## 图片已放在仓库中，打包下载即可
# 配置实验环境
- python
- imutils==0.5.4
- opencv-python==4.10.0.84
## 可运行以下命令自动安装
```bash
pip install -r requirements.txt
```
## 实验代码
```bash
# 导入库
import cv2
import argparse
import imutils
from imutils.perspective import four_point_transform

# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
args = vars(ap.parse_args())

# 加载图像
image_path = "sudoku.png"
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"找不到叫作 {image_path} 的图片文件.")

# 显示原始图像
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# 图像预处理：灰度化和高斯模糊
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 3)

# 自适应阈值化
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
# 反转二值图像
thresh = cv2.bitwise_not(thresh)

# 显示阈值化后的图像
cv2.imshow("Threshold Image", thresh)
cv2.waitKey(0)

# 轮廓检测
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# 排序轮廓，按面积从大到小
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

puzzleCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 检查是否是四边形轮廓
    if len(approx) == 4:
        puzzleCnt = approx
        break

# 如果没有找到四边形轮廓，则抛出错误
if puzzleCnt is None:
    raise Exception("找不到 {image_path} 的轮廓.")

# 在原图上绘制轮廓以验证结果
output = image.copy()
cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
cv2.imshow("Puzzle Outline", output)
cv2.waitKey(0)

# 透视变换
puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

# 显示透视变换后的图像
cv2.imshow("Puzzle Transform", puzzle)
cv2.waitKey(0)

cv2.destroyAllWindows()

```
# 思考题
- 为什么要进行预处理？
<br> 图像预处理的目的是提高图像质量，去除噪声和不必要的干扰，使得后续的图像分析更加精确。

- OpenCV 的自适应阈值处理函数 cv2.adaptiveThreshold 的原理是什么？<br>
[python3 opencv 图像二值化笔记（cv2.adaptiveThreshold）](https://blog.csdn.net/laoyezha/article/details/106445437)

- OpenCV 的轮廓提取函数 cv2.findContours 的调用语法是什么？<br>
[opencv学习—cv2.findContours()函数讲解（python）](https://blog.csdn.net/weixin_44690935/article/details/109008946)

- imutils 库中的 four_point_transform 函数，与透视变换函数有无区别？<br>
<br> four_point_transform 是 imutils 库中的一个函数，简化了透视变换的操作。与 OpenCV 的 cv2.getPerspectiveTransform 和 cv2.warpPerspective 组合使用不同，它只需要输入四个角点，自动计算透视矩阵并执行变换，简化了过程。
<br> 透视变换总结：
<br> cv2.getPerspectiveTransform：计算透视变换矩阵。
<br> cv2.warpPerspective：应用透视变换。
<br> imutils.four_point_transform：简化透视变换过程，自动计算和应用变换。
