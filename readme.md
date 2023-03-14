# Landmark-Retrieval-Based-on-Resnet


基于resnet的地标检索，初步检索之后可以使用基于标签的重排序方法改善结果。

## 环境

Python >= 3.6（3.6,3.7均适配）

torch ==1.4.0

tensorboardX==2.5等依赖包。

或者直接安装依赖的环境。

```
pip install -r requirements.txt
```



## 代码结构

```
|-- Landmark-Image-Retrieval
    |-- finetune  #微调resnet网络
    |  |-- paris  #数据集
    |  |-- runs   #训练的loss曲线
    |  |-- output #训练得到的模型
	|  |-- train_1.py #微调训练使用
	|  |-- train_2.py #顺序多数据集训练时使用
	|-- retrain #测试微调后的resnet网络的检索性能
	|  |-- vectors #存放预先计算好的索引集特征集合
	|  |-- output  #检索结果
	|  |-- metadata #数据集
	|-- resnetxxx #测试resnet预训练网络的检索性能
	|  |-- ...同上
	|-- source #测试resnet152预训练网络的检索性能和VLAD检索性能
	|  |-- ...同上
```

## 数据

我选择了[Paris6k](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)、[oxbuild](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)和自己构建的基于台北地标的数据集用于训练。

数据集结构如下：

```
|-- datasets
	|-- class_1
	|-- class_2
	...
	|-- class_n
```

使用以下命令划分数据集为训练集、验证集、测试集、索引集。

默认划分比例为：60%、10%、20%、10%

```
python3 main.py --mode create_dataset_csv --dataset paris #创建数据集的csv
python3 main.py --mode divide_dataset --dataset paris #划分数据集
```

如果需要修改划分比例，在preprocess.py中的divide_class修改。



## 训练

训练时数据集格式如下：

```
|-- datasets
	|-- train
	|  |-- class_1
	|  |-- class_2
	...
	|  |-- class_n
	|-- val
	|  |-- class_1
	|  |-- class_2
	...
	|  |-- class_n
```



训练命令：

```python
python train_1.py --data paris -b 32 --arch resnet152 --lr 0.001 --epochs 100 
```

1. --data  数据集名称

2. -b batchsize大小 

3. --arch 模型结构，默认为152

4. --lr 学习率

5. --epoch 指定在多少个epoch停止训练



optional arguments:
  -h, --help            show this help message and exit
  --data DIR            path to dataset
  --arch ARCH, -a ARCH  model architecture: (default: resnet152)
  -j N, --workers N     number of data loading workers (default: 4)  #几张卡
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --finetune            fine tune pre-trained model



训练的loss曲线将存在：```runs```中，类似格式```Apr25_11-35-57_ubunturesnet_paris```存储。

可以通过以下命令观察单次训练的loss曲线：

```
tensorboard --logdir=runs/xxx --port=6005 --bind_all
```

命令只输入runs可以同时观察多次训练的loss曲线。



所有checkpoint文件和best_model模型会保存到```output```中，格式如下：```2022-04-24_08:05:19_paris_165```. 最后的数字表示训练时的微调策略。

checkpoint文件每5个epoch保存一次。如果想在checkpoint文件上继续训练时使用```--resume```来读入。



## 测试

**修改检索的模型**

model.py  FeatureExtractor 中的self.res_model

修改为预训练的ResNet模型：

```
self.res_model = ResidualNet()
```

修改为微调过的模型：

model.py 开头的DIR修改模型路径。

然后修改此处的类别，修改为微调模型的训练的类别：

```
self.res_model = FineTuneModel(original_model, 'resnet152', 56)
```

**测试流程**：

1. 获取索引集的特征集合

```
python3 main.py --mode train --model resnet --dataset paris_index
```

2. 地标分类预测（分类）

   ```
   python3 main.py --mode predict --model resnet --dataset paris_index --query metadata/paris_test/eiffel/paris_eiffel_000012.jpg --distance L1
   ```

3. 地标检索（返回检索结果，默认检索序列为5）

   未重排序：

   ```
   python3 main.py --mode cbir --model resnet --dataset paris_index --query metadata/paris_test/eiffel/paris_eiffel_000012.jpg --distance L1 --method one
   ```

   使用重排序方法：

   ```
   python3 main.py --mode cbir --model resnet --dataset paris_index --query metadata/paris_test/eiffel/paris_eiffel_000012.jpg --distance L1 --method two
   ```

   检索效果分为两种：

   将检索结果和检索结果的路径存放在```output```中或者直接输出下图的result.jpg

   ![result](C:\Users\Alice\Desktop\result.png)

   选择不同具体检索结果的效果，需要修改main.py中对应的函数，具体函数都在prediction_metrics.py中，可以根据需要进行修改。

4. 指标计算

   **accuracy**

   ```
   python3 main.py --metric accuracy --model resnet --dataset paris_index --test_dataset metadata/paris_test --distance L1
   ```

   **mAP**

   ```
   python3 main.py --metric map --model resnet --dataset paris_index --distance L1 --method one
   python3 main.py --metric map --model resnet --dataset paris_index --distance L1 --method one
   ```

如果要测试SIFT+VLAD，将model修改为vlad即可。

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to your dataset
  --test_dataset TEST_DATASET   Path to your test dataset
  --query QUERY         Path to your query: example.jpg
  --model MODEL         vlad: Predict class using VLAD + SIFT

resnet: Pre-trained ResNet
  --mode MODE           train: Extract feature vectors from given dataset,
                        predict: Predict a class of query divide_dataset:
                        Dividing dataset into train, val, test and index from
                        dataset create_dataset_csv: Creating csv for dataset
                        cbir: Print 5 most similar photo for query vlad:
                        Predict class using VLAD + SIFT
  --metric METRIC       map: Calculate an average precision of dataset
                        accuracy: Calculate accuracy of prediction on testing
                        dataset
  --distance DISTANCE   L1: Manhattan Distance L2: Euclidean distance L3:
                        cosine similiarity #此处L1效果最好
  --method METHOD       one: origin two: rerank



## Reference

https://github.com/GrigorySamokhin/Landmark-Image-Retrieval
