# 介绍

welcome to learning coding

i am very excited as i have started to learn code

i was tying Tims all about it and he asked what is coding?

this is very good question and the very same thing that i had asked mum yeasterday

before we started/

my mum told me coding is a term for computer programming, it is how you give a computer

an instruction 

i have also discoverd that there are different coding language.

programmer use one or more  computer coding languages to get these instructions to computers

give example, programmers working at google might use C++ java c# javesvript or python

欢迎来到酷德青少年编程。我们不仅仅教你编程，更想要你像程序员一样，充满逻辑的思考一切问题！将想法有逻辑性的表达出来！

# 1. image labeling

注意：

开发环境

python3.8 tensorflow2.7

```
pip install labelImg
```

label each image which will get xml file. 

```shell
labelImg
```

The xml file contains image info such as width,length,bounding box coordinates.

# 2. xml to csv

convert xml file to csv file for training and testing
python xml_to_csv
RANDOM csv file

# 3. generate tfrecord

get train and test tfrecord which contains training data and testing data separately.

```
python generate_tfrecord.py --csv_input=images\train_labels.csv  --image_dir=images\train  --output_path=train.record

python generate_tfrecord.py --csv_input=images\test_labels.csv  --image_dir=images\test  --output_path=test.record
```



# 4. config label_map and training

### 4.1 modify label_map.pbtxt and download model from model zoo(get pipeline.config files)



### 4.2 modify pipeline.config file

### 4.3配置object_deteciton api

linux配置方式：

```linux shell
# Clone the tensorflow models repository
!git clone --depth 1 https://github.com/tensorflow/models
```

```
%%bash
sudo apt install -y protobuf-compiler
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

window配置方式：

```
# Clone the tensorflow models repository!git clone --depth 1 https://github.com/tensorflow/models
```

下载protobuf，版本要对。

https://github.com/google/protobuf/releases 
下载对应的版本系统在research文件夹下，执行命令提示符
<你的下载包路径位置(解压到当前路径下)>/bin/protoc  object_detection/protos/*.proto --python_out=.

复制object_detection/packages/tf2/setup.py 到objectdetion下

python -m pip install ./或者

python setup.py build

python setup.py install

在site-packages下看到object detection包，pycharm库中亦可以查看到该包，版本为0.1

注：还不能导入查看是否有冲突文档，还不行增加环境变量







ps:回办公室拿来了“超级电脑”顶替之前家里的gtx 950（感觉可以使用google colab的gpu）

重新setup object_detection api 之后（又是一顿的报错，直接卸载了原来的开发环境，
现改为python3.8，cuda11.1（这里也报错，推荐我去下载cudnn11.2去支持）

再次修改pipeline.config
batchsize 改为1（看你内存大小）（我改为6都给我报错，说我用完了oom，看来也不是超级电脑啊。。后来发现4可以。）

### 再次更改model_main_tf2 ,开头加上GPU物理使用情况。

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:

Restrict TensorFlow to only allocate 1GB of memory on the first GPU

  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:

   	Virtual devices must be set before GPUs have been initialized

​    	print(e)

```shell
python model_main_tf2.py --alsologtostderr  --model_dir=pretrainedmodel  --pipeline_config_path=D:\pycharm\TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master\training\pipeline.config
```

目前在训练的样子。。我要看看你要训练多久。我都给你录频了。
3 hours later。。。（出门干事回来发现好了）

# 5. 可视化训练结果

```
tensorboard --logdir=pretrainedmodel
```



# 6. export model

Example Usage:

python exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_dir path/to/checkpoint \
    --output_directory path/to/exported_model_directory
    --use_side_inputs True/False \
    --side_input_shapes dim_0,dim_1,...dim_a/.../dim_0,dim_1,...,dim_z \
    --side_input_names name_a,name_b,...,name_c \
    --side_input_types type_1,type_2

```
e.g. python exporter_main_v2.py --input_type image_tensor  --pipeline_config_path D:\pycharm\TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master\training\pipeline.config  --trained_checkpoint_dir D:\pycharm\TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master\pretrainedmodel --output_directory D:\pycharm\TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master\exportmodel01
```

这检测器不是很理想呢。

重新修改并完善吧



# 7. export tflite

*`only ssd model support now`*

first, you need to export model as a pre-tflite tensorflow model

please use exporter_graph_tf2.py to export model. 

```
python exporter_graph_tf2.py --input_type image_tensor  --pipeline_config_path D:\pycharm\TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master\training\pipeline.config  --trained_checkpoint_dir D:\pycharm\TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master\pretrainedmodel --output_directory D:\pycharm\TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master\tflitemodel
```

python tfconverttflite.py



# 8.training in google cloud

upload files to google drive

setup environment

note: colab cuda runtime is 8.0.5 but compiled with 8.1.0

```
%%bash

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin

sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb

sudo apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub

sudo apt-get update

sudo apt-get -y install cuda
```



```
!pip install tensorflow==2.7
!pip install keras==2.7
```

加载google drive

```
from google.colab import drive

drive.mount('/content/gdrive')
```

加载对应的cudnn版本，对应之前的cuda。

```
!tar -zvxf /content/gdrive/MyDrive/cudnn-11.2-linux-x64-v8.1.0.77.tgz
```

```
%%bash
cd cuda/include
sudo cp *.h /usr/local/cuda/include/

%%bash
cd cuda/lib64
sudo cp lib* /usr/local/cuda/lib64/
```

安装object——detection api

```
# Clone the tensorflow models repository

!git clone --depth 1 https://github.com/tensorflow/models
```

```
%%bash

sudo apt install -y protobuf-compiler

cd models/research/

protoc object_detection/protos/*.proto --python_out=.

cp object_detection/packages/tf2/setup.py .

python -m pip install .
```

可能会出现numpy和opencv-python报错，可执行下面代码解决

```
!pip install "opencv-python-headless<4.3"

!pip3 install numpy --upgrade
```

切换到程序路径下

```
%cd /content/gdrive/MyDrive/object_detection01
```

```
!python model_main_tf2.py --alsologtostderr  --model_dir=pretrainedmodel  --pipeline_config_path=pipelinecloud.config
```

执行和之前一样的操作，再colab中gpu速度明显比再local machine的gpu快很多，接下来应该操作和之前一样了；

待整理一个完整的环境。