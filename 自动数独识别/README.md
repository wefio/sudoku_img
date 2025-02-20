# OCR（Optical Character Recognition）简介
OCR 是一种将印刷文本或手写文本从图像转换为可编辑和搜索的文本的技术。它广泛应用于文档扫描、证件识别、车牌识别和数字化存档等场景，极大地提高了信息处理效率。
## 1.	Pytesseract 和 Tesseract-OCR
Pytesseract 是一个 Python 包，它作为开源工具 Tesseract-OCR 的封装接口，由 Hewlett Packard 实验室开发并在2005年实行开源；自2006年后，谷歌接手并联合优秀的开源贡献者持续维护和发展。Tesseract 在3.x版本后逐渐成熟，不仅支持多种图片格式，还加入了多语言文本识别功能。尽管3.x版本仍基于传统计算机视觉算法，但得益于近年来深度学习技术的迅猛发展，Tesseract 4.0 版本引入了基于 LSTM（长短期记忆网络，一种循环神经网络）的深度学习模块，这使得其在准确率和速度方面有了显著提升。
## 2.	实验环境
操作系统：Windows 11
编程语言：Python 3.12
依赖库：
  pytesseract: 用于调用 Tesseract OCR 引擎进行文字识别。
  opencv-python: 提供图像处理功能，如灰度化、二值化、去噪等预处理步骤。
  其他依赖项详见 requirements.txt 文件。
Tesseract 版本：使用的是 tesseract-ocr-w64-setup-5.5.0.20241111 安装包。
## 3.	环境安装

为了设置工作环境，请遵循以下步骤：

安装 Tesseract 工具：根据提供的链接下载并安Tesseract https://github.com/tesseract-ocr/tesseract
安装 Pytesseract：利用 pip 工具来安装 Pytesseract 包：
  bash
  pip install pytesseract
## 4.	运行步骤
本项目推荐使用 Visual Studio Code (VSCode) 并结合 Conda 虚拟环境进行开发。以下是运行指南：
### 1. 将包含代码的工作文件夹拖入 VSCode 中。
### 2. 使用命令行导入所有依赖：
```bash
   pip install -r requirements.txt
```
### 3. 设置 Tesseract 可执行文件路径，确保该路径正确无误。
### 4. 配置数独图像路径，项目已提供6个示例图片，这些样本经过测试，具有良好的识别效果。
### 5.	使用到的核心算法和技术
预处理：通过 OpenCV 库对图像进行初步处理，包括灰度化、自适应二值化以及形态学操作去除噪声，以提高后续处理的质量。
数独框检测：采用轮廓检测与多边形逼近方法找到并确定数独表格的边界。
透视变换与裁剪：运用透视变换校正图像，使其成为标准矩形，并通过裁剪去除无关边缘部分。
图像增强：结合高斯二值化与中值滤波进一步优化字符清晰度，减少干扰因素。
OCR 识别：借助 Pytesseract 实现从图像中提取数字的功能，并利用多线程加速整个识别过程。
数独求解：实现了回溯法算法来解决数独问题，即通过递归方式尝试填充每个空格直至找到完整解决方案或证明无解。
6.	参考
在线数独网站
https://www.sudoku.name/index-cn.php
https://htlsmile.github.io/2020/03/08/%E6%95%B0%E7%8B%AC%E7%9A%84%E8%AF%86%E5%88%AB%E4%B8%8E%E6%B1%82%E8%A7%A3/
http://zh.sudoku.menu/zcheckuj.html
程序参考
https://blog.csdn.net/weixin_72749746/article/details/139843761
https://github.com/ChaosJulien/XiaoYuanKouSuan_Auto/blob/main/%E5%B0%8F%E7%8C%BF%E6%90%9C%E9%A2%98.py
解数独算法
https://leetcode.cn/problems/sudoku-solver/
程序由ChatGPT辅助完成

## 一些发现
tesseract似乎对衬线字体的数字识别正确率更高
无衬线体 黑体123456789
衬线体Times New Roman 1234567890
无衬线体的1容易识别为7
