数据集扩充增广/格式转换

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1ebbbe1ca9d94ddfb92bafb25b248f74.png)


适用于yolov3，v5，v7，v8，v9，v10，v11，v12目标检测数据集，支持同步更新标签（txt）

小工具，Windows10以上系统可以直接打开使用，无需安装，无需配置环境![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d3f34bc389a94df2b4eebac815331830.png)


翻转、旋转、缩放、拉伸、剪裁、平移、亮度、对比度、饱和度、噪声等十多种增广方式

可选择增广倍数，生成的图像和标签重命名不会覆盖原数据

新增格式转换（XML→TXT，TXT→XML）和划分数据集功能![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/49d791df406342cd94f212a7b960fdf4.png)


附带有使用说明和演示视频
使用了Python的`OpenCV`进行图像处理，`xml.etree.ElementTree`处理VOC格式，以及`shutil`进行文件操作。

### 核心功能代码框架 (Python)

同学你可以使用以下代码作为基础，配合一个简单的GUI（如PyQt5）来构建这个工具。

#### 1. 环境依赖
```bash
pip install opencv-python numpy lxml
```

#### 2. 数据增广核心逻辑 (Augmentation Logic)

```python
import cv2
import numpy as np
import os
import random
import shutil

class DataAugmentor:
    def __init__(self, img_path, label_path, output_dir):
        self.img = cv2.imread(img_path)
        self.label_path = label_path
        self.output_dir = output_dir
        self.h, self.w = self.img.shape[:2]
        
        # 假设是YOLO格式标签 (class_id, x_center, y_center, width, height)
        self.boxes = self.load_yolo_labels(label_path) 

    def load_yolo_labels(self, path):
        boxes = []
        if not os.path.exists(path): return boxes
        with open(path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                boxes.append([int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4])])
        return boxes

    def save_yolo_labels(self, boxes, save_path):
        with open(save_path, 'w') as f:
            for box in boxes:
                # box format: [class_id, x_c, y_c, w, h]
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

    # --- 增广方法 ---

    def random_brightness(self, img, delta=0.2):
        # 随机亮度调整
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = v.astype(np.float32)
        v *= (1 + random.uniform(-delta, delta))
        v = np.clip(v, 0, 255).astype(np.uint8)
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def random_flip(self, img, boxes):
        # 左右翻转
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            new_boxes = []
            for b in boxes:
                # x_center = 1 - x_center
                new_boxes.append([b[0], 1 - b[1], b[2], b[3], b[4]])
            return img, new_boxes
        return img, boxes

    def random_scale(self, img, boxes, scale_range=(0.8, 1.2)):
        # 随机缩放
        scale = random.uniform(*scale_range)
        new_w, new_h = int(self.w * scale), int(self.h * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # 调整回原尺寸 (可选，或者保持缩放后尺寸)
        img = cv2.resize(img, (self.w, self.h))
        
        # 更新标签 (简单缩放示例，实际需考虑裁剪或填充)
        new_boxes = [[b[0], b[1], b[2], b[3]*scale, b[4]*scale] for b in boxes]
        return img, new_boxes

    def process(self, save_name):
        # 组合增广
        img_aug = self.img.copy()
        boxes_aug = self.boxes.copy()

        # 示例：执行亮度调整
        if random.random() > 0.5:
            img_aug = self.random_brightness(img_aug)
            
        # 示例：执行翻转
        img_aug, boxes_aug = self.random_flip(img_aug, boxes_aug)

        # 保存
        cv2.imwrite(os.path.join(self.output_dir, f"{save_name}.jpg"), img_aug)
        self.save_yolo_labels(boxes_aug, os.path.join(self.output_dir, f"{save_name}.txt"))

# 使用示例
# aug = DataAugmentor('test.jpg', 'test.txt', './output')
# aug.process('augmented_001')
```

#### 3. GUI 界面构建 (使用 PyQt5 示例)

截图中的界面非常经典，你可以用 PyQt5 快速搭建：

```python
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QComboBox, QMessageBox

class AugmentTool(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('YOLO数据集增广工具')
        self.setGeometry(300, 300, 800, 600)

        # 布局
        layout = QVBoxLayout()

        # 文件选择区
        h_layout1 = QHBoxLayout()
        self.btn_img = QPushButton('选择图片目录')
        self.btn_img.clicked.connect(self.select_img_dir)
        h_layout1.addWidget(self.btn_img)
        
        self.btn_label = QPushButton('选择标签目录')
        self.btn_label.clicked.connect(self.select_label_dir)
        h_layout1.addWidget(self.btn_label)
        
        layout.addLayout(h_layout1)

        # 增广方式选择
        self.combo_method = QComboBox()
        self.combo_method.addItems(['随机亮度', '随机翻转', '随机缩放', '混合增广'])
        layout.addWidget(QLabel('选择增广方式:'))
        layout.addWidget(self.combo_method)

        # 执行按钮
        self.btn_start = QPushButton('开始增广')
        self.btn_start.clicked.connect(self.start_augmentation)
        layout.addWidget(self.btn_start)

        # 显示区域 (截图左侧的预览)
        self.image_label = QLabel('图像预览区域')
        self.image_label.setStyleSheet("background-color: #f0f0f0; height: 400px;")
        layout.addWidget(self.image_label)

        self.setLayout(layout)

    def select_img_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        # 这里需要添加逻辑去遍历文件夹并显示图片

    def select_label_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "选择标签文件夹")

    def start_augmentation(self):
        # 调用上面的 DataAugmentor 类进行批处理
        QMessageBox.information(self, "提示", "增广任务已启动！")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AugmentTool()
    ex.show()
    sys.exit(app.exec_())
```

### 这个工具的关键技术点解析

1.  **同步更新标签**：这是最核心的难点。
    - 当你做**几何变换**（如翻转、平移、缩放）时，图像的像素变了，标签框的坐标（x, y, w, h）也必须跟着变。
    - 代码中需要建立图像坐标与YOLO归一化坐标之间的映射关系。
2.  **格式转换 (XML ↔ TXT)**：
    - **XML to TXT**：解析`<object>`标签中的`<bndbox>`坐标，根据图片宽高归一化，转换为YOLO格式。
    - **TXT to XML**：反向操作，将归一化坐标还原为像素坐标，填入XML模板。
3.  **免安装打包**：
    - 开发者通常会使用`PyInstaller`将上述Python脚本打包成`.exe`文件。
    - 命令示例：`pyinstaller -F -w your_script.py`（`-w` 表示去除控制台黑窗口，`-F` 表示打包成单文件）。
根据图片中的文件列表（`.exe`可执行文件、视频教程、使用说明）以及之前的描述，这个工具显然是一个打包好的Python应用程序。

要构建一个类似图片中`目标检测txt增强工具_20250516.exe`的软件，核心在于使用Python编写逻辑，并使用`PyQt5`或`Tkinter`制作界面，最后通过`PyInstaller`打包。

下面是一个精简版的核心源码框架。这个代码实现了截图中提到的核心功能：**图像增广（翻转、亮度等）**以及**同步更新YOLO格式的TXT标签**。

### 核心代码实现（Python + OpenCV）

你可以将以下代码保存为 `main.py`。

```python
import cv2
import os
import numpy as np
import random
import shutil

class YOLODataAugmentor:
    def __init__(self, img_path, txt_path, output_img_dir, output_txt_dir):
        """
        初始化增广工具
        :param img_path: 原图路径
        :param txt_path: 原标签(txt)路径
        :param output_img_dir: 增强后图片保存目录
        :param output_txt_dir: 增强后标签保存目录
        """
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise ValueError(f"无法读取图像: {img_path}")
            
        self.h, self.w = self.img.shape[:2]
        self.txt_path = txt_path
        self.output_img_dir = output_img_dir
        self.output_txt_dir = output_txt_dir
        
        # 读取YOLO格式标签 (class_id, x_center, y_center, width, height)
        self.boxes = self._load_labels(txt_path)

    def _load_labels(self, path):
        """读取txt标签文件"""
        boxes = []
        if not os.path.exists(path):
            return boxes
        with open(path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    # 转换为浮点数存储
                    boxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        return boxes

    def _save_labels(self, boxes, new_txt_path):
        """保存YOLO格式标签"""
        with open(new_txt_path, 'w') as f:
            for box in boxes:
                # 格式化写入，保留6位小数
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

    def _yolo_to_pixel(self, box):
        """将YOLO归一化坐标转换为像素坐标"""
        # box: [class_id, x_c, y_c, w, h]
        x_c = box[1] * self.w
        y_c = box[2] * self.h
        w = box[3] * self.w
        h = box[4] * self.h
        
        x1 = int(x_c - w / 2)
        y1 = int(y_c - h / 2)
        x2 = int(x_c + w / 2)
        y2 = int(y_c + h / 2)
        return [box[0], x1, y1, x2, y2]

    def _pixel_to_yolo(self, box_pixel):
        """将像素坐标转换为YOLO归一化坐标"""
        # box_pixel: [class_id, x1, y1, x2, y2]
        cls_id = box_pixel[0]
        x1, y1, x2, y2 = box_pixel[1], box_pixel[2], box_pixel[3], box_pixel[4]
        
        # 防止坐标越界
        x1 = max(0, min(x1, self.w))
        x2 = max(0, min(x2, self.w))
        y1 = max(0, min(y1, self.h))
        y2 = max(0, min(y2, self.h))

        if x2 <= x1 or y2 <= y1: return None # 无效框

        w = x2 - x1
        h = y2 - y1
        x_c = x1 + w / 2
        y_c = y1 + h / 2
        
        return [cls_id, x_c / self.w, y_c / self.h, w / self.w, h / self.h]

    def flip_horizontal(self, save_name="flip"):
        """水平翻转 (镜像)"""
        # 1. 图像翻转
        img_flipped = cv2.flip(self.img, 1)
        
        # 2. 标签坐标变换
        new_boxes = []
        for box in self.boxes:
            cls_id, x1, y1, x2, y2 = self._yolo_to_pixel(box)
            # 新的x坐标 = 图像宽度 - 原来的x坐标
            new_x1 = self.w - x2
            new_x2 = self.w - x1
            new_box = self._pixel_to_yolo([cls_id, new_x1, y1, new_x2, y2])
            if new_box:
                new_boxes.append(new_box)
        
        # 3. 保存
        self._save_results(img_flipped, new_boxes, save_name)

    def adjust_brightness(self, beta=50, save_name="bright"):
        """调整亮度 beta: 亮度增量 (-100 到 100)"""
        # 1. 图像调整
        # new_img = alpha * img + beta
        new_img = cv2.convertScaleAbs(self.img, alpha=1.0, beta=beta)
        
        # 2. 标签不变，直接复制
        self._save_results(new_img, self.boxes, save_name)

    def _save_results(self, img, boxes, suffix):
        """辅助函数：保存图片和对应的txt"""
        # 生成随机文件名或按规则命名，防止覆盖
        # 这里简单使用时间戳或随机数演示
        uid = str(random.randint(1000, 9999))
        img_name = f"{suffix}_{uid}.jpg"
        txt_name = f"{suffix}_{uid}.txt"
        
        cv2.imwrite(os.path.join(self.output_img_dir, img_name), img)
        self._save_labels(boxes, os.path.join(self.output_txt_dir, txt_name))
        print(f"生成: {img_name}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 模拟路径
    img_path = "test.jpg"      # 你的原图
    txt_path = "test.txt"      # 你的原标签
    out_img = "output/images"
    out_txt = "output/labels"
    
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_txt, exist_ok=True)
    
    # 实例化
    aug = YOLODataAugmentor(img_path, txt_path, out_img, out_txt)
    
    # 执行增广
    aug.flip_horizontal(save_name="flip_h")      # 水平翻转
    aug.adjust_brightness(beta=40, save_name="bright_up") # 变亮
```

### 如何变成截图中的 `.exe` 软件？

是使用 **PyQt5** 或 **PySide6** 开发的。要复刻那个软件，你需要做以下几步：

1.  **开发GUI界面**：
    使用 `PyQt5` 编写界面，包含“选择文件夹”按钮、“增广倍数”输入框、“开始增广”按钮等。
2.  **整合逻辑**：
    将上面的 `YOLODataAugmentor` 类整合进GUI的逻辑中，遍历用户选择的文件夹下的所有图片进行批量处理。
3.  **打包成EXE**：
    使用 `PyInstaller` 将Python脚本打包成独立的 `.exe` 文件，这样在没有安装Python的电脑上也能运行

**打包命令示例：**

```bash
# 安装打包工具
pip install pyinstaller

# 打包命令 (-F生成单文件, -w隐藏控制台窗口, -i添加图标)
pyinstaller -F -w -i icon.ico main.py
```

### 总结
使用上面的Python脚本，或者寻找开源的类似工具（如 `Albumentations` 库结合 YOLO 脚本）。

以上文字及代码仅供参考。
。
