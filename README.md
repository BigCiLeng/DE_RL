# 同济大学船舶设计强化学习部分

#### 介绍

本仓库是船舶设计项目同济大学负责的强化学习部分，旨在利用前面模糊推理和最小凸包获得的船舶参数来进一步设计并获得较优的船舶参数。

#### 软件架构

+ `algorithm`：核心的强化学习算法的实现代码文件夹，里面可以有多个算法

  + `log`：用来存放tensorboard产生的各种数据的文件夹，可以用来查看运行状况。

    ```shell
    cd log # 切换到log目录下
    conda activate tianshou # 激活自己的环境
    tensorboard --dirlog=dqn # 调用想要查看的算法的运行状况
    ```
    
  + `DQN.py`：实现的最简单的DQN算法
  
  + `test`：存放测试代码的文件夹
  
+ `back_up`：以前尝试的各种代码的实现

+ `data_input`：数据读取的代码存放的文件夹，有两种其中一种为联调，一种为从文件中读取

  + `input_from_file.py`：从mat文件中读取数据，主要用于本地调试代码
  
+ `data_preprocess`：预处理代码存放的文件夹

  + `preprocess.py`：对数据进行预处理的代码，分别是处理数据的主函数data_solve，排序函数rank，以及归一化函数normalize
  + `rank_eva.py`：读取csv文件获取用户输入的排序。 #TODO
  + `weights_matrix`：最小二乘法实现根据用户排序优化权重
  
+ `data_source`：该文件夹为保存收集的数据，保存格式为mat

+ `env_design`：强化学习的环境设计，预计设计成两种。场景为在高维空间中搜索点的位置。

  + `env.json`：环境的配置，有一些需要调节的参数在这里面。
  + `env1.py`：单点环境，即仅输入一个点作为圆心来进行搜索。
  
+ `config.json`：总体的配置文件，里面有文件的读取路径以及进程数等

+ `evaluate.py`：评价函数，调用动态链接库获得性能计算指标。根据性能指标和得分曲线获得得分

+ `main.py`：训练时调用

+ `main_test.py`：测试时调用

+ `main_weights.py`：权重优化测试时调用

#### 环境配置教程

1. 使用`requirments.txt`来安装python环境

   ```shell
   pip install -r requirments.txt
   ```

   > 如果安装出错，则去`requirments.txt`中删除对应的包
   >
   > pytorch的安装：`requirments.txt`中是cuda11.3，请修改来安装对应版本

2. 查看`评估函数配置.md`安装后续配置

3. 如果有需求在linux中安装matlab，则看这个[博客]([docker中命令行安装Matlab2021a的Linux版本 | skw的小站 (skwyx544818lh.ml)](https://skwyx544818lh.ml/post/docker-zhong-ming-ling-xing-an-zhuang-matlab2021a-de-linux-ban-ben/))

   

#### 使用说明

1. 训练时调用和修改`main.py`，测试时调用和修改`main_test.py`，权重优化时调用和修改`main_weights.py`

#### 参与人员

1. 孙凯威
