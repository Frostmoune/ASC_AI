## 运行说明  
*注：原baseline详见我github中fork的另一个仓库neural-vqa-tensorflow*  
  
### 环境及依赖包  
- python3.6  
- tensorflow_gpu 1.4及以上  
- h5py  
- zltk  

### 训练前准备  
1. 在代码所在根目录新建Data文件夹  
2. 从VQA（版本为v1）中下载训练集、验证集（图片 + 问题 + 答案），放到Data文件夹中  
3. 运行data_loader.py  
4. 下载不同的image embedding模型：
    - 从https://github.com/ry/tensorflow-vgg16/raw/master/vgg16-20160129.tfmodel.torrent 下载VGG16，
    并将vgg16.tfmodel保存到Data文件夹中  
    - 从http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz 下载ResNet101，
    并将解压缩得到的resnet_v2_101.ckpt保存到Data/ResNet中  
    - 从http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz 下载Inception_v3，
    并将解压缩得到的ckpt文件改名为inception_v3.ckpt，保存到Data/Inception中  
5. 得到训练集图片特征：
   - `python extract_fc7.py`得到VGG16的特征  
   - `python extract_resnet.py`得到ResNet101的特征  
   - `python extract_inception.py`得到Inception_v3的特征  
   *注：若要得到验证集图片的特征，需要加上`--split val`的选项*  
6. 新建Data/Logs和Data/Models文件夹  

### 训练  
- 直接执行`python train.py`即可  
- 选项(各选项含义详见https://github.com/paarthneekhara/neural-vqa-tensorflow)：
    - 补充选项：
    image_features:选用的特征种类（可选`vgg16`、`resnet101`和`inception_v3`，默认为vgg16）
    *注：当选择vgg16以外的模型时，feature_length需设为2048（默认为4096）*  
- 训练示例：
  `python train.py --feature_length 2048 --image_features resnet101 --learning_rate 0.001 --epochs 10 --num_lstm_layers 2` 

### 验证  
- 验证前需准备好验证集的图片特征  
- 执行`python evaluate.py`  
- 需要使用与之前训练过程中一样的参数，并用`model_path`参数指定模型路径  
- 验证示例：
  `python evaluate.py --feature_length 2048 --image_features resnet101 --learning_rate 0.001 --epochs 10 --num_lstm_layers 2 --model_path Data/Models/model7.ckpt`  

### 测试（预测）前准备  
* 注1：这里直接使用VQA_v1的测试集和测试问题进行测试，建议在Window下进行  
* 注2：测试部分需要用到Pillow这个包    
* 注3：为了节省时间，只用了测试集前100张图进行测试  
1. 从VQA_v1中下载测试集和测试问题，放到Data文件夹中  
2. 注释data_loader.py的321行，解除322行的注释，然后运行`python data_loader.py`  
3. 同训练前准备步骤4
4. 同训练前准备步骤5，加上`--split val`的选项 

### 测试（预测）  
- 执行`python test.py`  
- 测试部分所用参数与验证部分参数种类相同  
- 测试示例：
  `python test.py --feature_length 2048 --image_features resnet101 --num_lstm_layers 2 --model_path Data/Models/model7.ckpt`  
- 每测试一个问题，会显示问题并弹出问题所对应的图片；每测试完一个问题，可输入回车继续测试  
- 输入ctrl + c可退出测试  