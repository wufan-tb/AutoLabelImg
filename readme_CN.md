# AutoLabelImg 多功能自动标注工具

![AutoLabelImg](./demo/demo.png)

### [<u>English</u>](./readme.md)    |    [<u>中文</u>](./readme_CN.md)

### 简介：

在[labelImg](https://github.com/tzutalin/labelImg)的基础上，增加了多种标注工具，放在**Annoatate-tools**和**Video-tools**两个菜单栏下面。具体功能包含如下：

- **`TOOL LIST`**：
- [x] **自动标注**：基于yolov5的模型自动标注
- [x] **追踪标注**：利用opencv的追踪功能，自动标注视频数据
- [x] **放大镜**：局部放大，对小目标的标注有帮助，可以关闭
- [x] **数据增强**：随机使用平移，翻转，缩放，亮度，gama，模糊等手段增强图片
- [x] **查询系统**：输入关键字获得详细说明信息
- [x] 其他辅助工具：类别筛选/重命名/统计、标注文件属性校正、视频提取/合成、图片重命名等，可以利用查询系统查看详细信息，欢迎体验

### Demo:

因文件较大，视频已上传至B站

[自动标注](https://www.bilibili.com/video/BV1Uu411Q7WW/)

[视频追踪标注](https://www.bilibili.com/video/BV1XT4y1X7At/)

[放大镜](https://www.bilibili.com/video/BV1nL4y1G7qm/)

[数据增强](https://www.bilibili.com/video/BV1Vu411R7Km/)

[查询系统](https://www.bilibili.com/video/BV1ZL4y137ar/)

### 更新日志：

2022.01.14：自动标注去掉Retinanet，仅保留yolov5，并增加标签选择

2022.01.11：优化放大镜卡顿现象，增加放大镜可关闭选项

2020.12.28：增加视频追踪标注工具

2020.12.10：初步把所有工具加进labelimg，版本1.0

## 安装步骤：

1. 复制仓库：

   ```bash
   git clone https://github.com/wufan-tb/AutoLabelImg
   cd AutoLabelImg
   ```

2. 安装依赖：

   ```bash
   conda create -n {your_env_name} python=3.7.6
   conda activate {your_env_name}
   pip install -r requirements.txt
   ```

3. 源码编译：

   **Ubuntu用户:**
   
   ```
   sudo apt-get install pyqt5-dev-tools
   make qt5py3
   ```
   
   **Windows用户:**
   
   ```
   pyrcc5 -o libs/resources.py resources.qrc
   ```
   
4. 准备yolov5模型并放置在如下位置，官方模型获取参考[Yolov5](https://github.com/ultralytics/yolov5)

   ```bash
   mv {your_model_weight.pt} pytorch_yolov5/weights/
   ```

5. 打开软件，开始标注

   ```
   python labelImg.py
   ```

## 设置快捷方式[非必须]

**Windows用户:**

桌面创建labelImg.bat（可以新建文本文件，然后把后缀.txt改成.bat）,右键用文本编辑器打开，键入下面内容(不一定是D盘，根据实际输入)：

```bash
D:
cd D:{path to your labelImg folder}
start python labelImg.py
exit
```

下面是一个实际案例，根据自己的实际路径修改第一二行即可：

```
D:
cd D:\_project\AutoLabelImg
start python labelImg.py
exit
```

双击labelImg.bat即可打开标注软件。

**Ubuntu用户:**

打开环境变量文件：

```bash
vim ~/.bashrc
```

然后增加下面内容：

```bash
alias labelimg='cd {path to your labelImg folder} && python labelImg.py
```

使环境变量生效：

```bash
source ~/.bashrc
```

然后在终端输入指令'labelimg'即可打开标注软件。

## 引用

```
{   AutoLabelImg,
    author = {Wu Fan},
    year = {2020},
    url = {\url{https://https://github.com/wufan-tb/AutoLabelImg}}
}
```

