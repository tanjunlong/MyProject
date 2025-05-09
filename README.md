# *segmentation of purple clay teapots*

## *一、<span id="jump">框架搭建</span>*
### 1. 模型选择
考虑到自身电脑的硬件性能有限，为了实现模型与设备的良好适配以及达成模型轻量化的目标，最终选择采用 UNet 分割模型，可以直接访问[UNet模型的GitHub下载网址][id]来直接下载预处理过后的UNet模型。


>1.1. 模型简介
The u-net is convolutional network architecture for fast and precise segmentation of images. Up to now it has outperformed the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. It has won the Grand Challenge for Computer-Automated Detection of Caries in Bitewing Radiography at ISBI 2015, and it has won the Cell Tracking Challenge at ISBI 2015 on the two most challenging transmitted light microscopy categories (Phase contrast and DIC microscopy) by a large margin (See also our annoucement).
	 ![UNet Model](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png  "示例1")
					u-net模型结构图
### 2. 本地所需环境的搭建
```
	PyTorch2.0.0 + Cuda1.7.0 + Python 3.8
    所需工具包
		① matplotlib==3.6.2
		② numpy==1.23.5
		③ Pillow==9.3.0
		④ tqdm==4.64.1
		⑤ wandb==0.13.5
注：项目文件中doucument/requirements.txt包含上面所需的工具包
```

## *二、数据集的收集以及预处理*
### 1. 数据收集
 在博物馆的紫砂壶展厅收集累计286张原始紫砂壶相关的高清图片，此外也在青瓷展厅采集了一些青瓷器图来辅助模型验证。示例紫砂壶合集图片如下：
 ![紫砂壶](https://raw.githubusercontent.com/tanjunlong/MyProject/master/imgShow/datasetEx.jpg  "示例")
 
 
### 2. 标注数据集
选择labelme软件进行标注时，有一点要注意，使用labelme软件时，可以通过手动多边形标注，也可以用它自带的AI模型进行标注，但是后者对电脑的性能要求较高。示例标注且转码后掩码图合集如下：
![标注图](https://raw.githubusercontent.com/tanjunlong/MyProject/master/imgShow/dataSetMasksEx.jpg  "示例")


>注：因为对图片处理完成之后生成了.json文件，所以我们直接编写py脚本利用labelme_export_json.exe对若干个json文件执行转换后生成对应的png格式图片。

## *三、模型训练*
### 1. 软硬件平台的选择
如果本地有性能较高的显卡，则本机训练就行，若是没有高性能的显卡，可以考虑在GPU算力平台上租赁相关的显卡平台进行训练。本人选择租赁的是3090(24G显存)显卡+AMD Ryzen 5 5600X,训练十轮，批次大小为1的情况下，训练时间大致在1小时左右。
### 2. 预定义模型
```python
	def train_model(
        model,
        device,
        epochs: int = 10,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
```
### 3. 调整训练参数
```python
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    return parser.parse_args()
```
>注：在get_args函数中调整epochs、batchSize、learningRate等超参数...

### 4. 训练过程
调整完合适的超参数过后，可以直接在命令行输入python train.py进入到训练过程，训练完成后会打印显示Iou、ACC、validationDice等信息。在执行的过程中，每一轮训练完成后会生成相应的模型权重文件(.pth文件)，也就是说多轮训练会产生多个模型权重文件，其默认放置在生成的checkpoints文件夹中，我们可以通过predict.py并结合checkpoint_epoch.pth文件来进行验证，例如：python predict.py -i data/imgs/1.jpg -m checkpoints/checkpoint_epoch.pth

## *四、实验结果*
不同超参数下的模型的F1-score
| 超参数 | F1-Score | Iou | ACC |
|:---:|:---:|:---:| :---:|
| **epochs：5<br>batch_size：1<br>learning_rate：1e^-5^** | 0.533 | 0.319 | 0.853 |
| **epochs：10<br>batch_size：3<br>learning_rate：3e^-4^** | 0.612 | 0.368 | 0.878 |
| **epochs：20<br>batch_size：5<br>learning_rate：1e^-5^** | 0.687 | 0.343 | 0.903 |

## *五、模型效果*
**测试图片合集示例：**
![标注图](https://raw.githubusercontent.com/tanjunlong/MyProject/master/imgShow/validation.jpg  "示例")
**测试效果合集示例：**
![标注图](https://raw.githubusercontent.com/tanjunlong/MyProject/master/imgShow/validationMasks.jpg  "示例")

## *六、项目相关文件说明*
>由于GitHub本地上传大小限制100MB，所以将数据集图片，掩码图片，.json文件压缩成上图的若干个压缩部分：<br>
		①code文件夹中包含所有程序代码，需将其中的code.rar压缩文件中包含的所有文件都解压到data同级目录下（如下图所示）<br>
		②data文件夹中data文件夹中的imgs文件夹中包含所有原始数据集的压缩包，需要将12个parts压缩包全部解压到data/imgs文件夹下；masks文件夹中包含所有原始数据集对应的掩码图，需将img_masks.rar压缩文件解压到data/masks文件夹下<br>
		③document文件夹中包含requirements.txt文件，里面包含所有所需的工具包说明<br>
		④imgShow仅放置GitHub仓库中的展示图片<br>
	![标注图](https://raw.githubusercontent.com/tanjunlong/MyProject/master/imgShow/contentStructure.png  "示例")
  [点击跳转至开头](#jump)<div>

[id]:https://github.com/zhixuhao/unet
