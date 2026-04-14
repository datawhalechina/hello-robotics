# Isaac Sim环境配置及安装

### hello-robotics整个教程基于：
* ### ubuntu 22.04
* ### ros2 humble 
* ### Isaac Sim5.1
* ### Isaac lab2.3.1

## 有效性：2026年4月14日。后续CUDA或者环境版本更新，请自行按照以下逻辑安装。尽量为大家详细的讲解每一个安装的细节和可能遇到的坑~，当然也欢迎大家补充~

## 最好按照以下顺序安装，否则有可能会出现安装完isaacsim后，安装ros2，出现isaacsim不能够自动适配ros2安装路径的问题。

linux实用修改权限命令推荐：

```python
sudo chmod 777 文件或文件夹名
```

写在最前面，安装所有的东西之前，一定要先更新你的所有软件包：

```python
sudo apt update
sudo apt upgrade
```

## 1.1 系统配置

### 已经安装过的可以跳过前往第二部分直接查看Isaacsim安装

### 1.1.1 NVIDIA显卡驱动检查

```python
nvidia-smi
```

输入以上命令检查驱动版本和GPU状态，应该输出如下示例：

<img src="assets\nvidia-smi.png"/>

请注意，驱动的版本一定要跟显卡匹配，建议通过系统的“软件和更新”来安装，当然，如果有其他更好的安装方法也可以使用，保证驱动版本和显卡匹配即可。

<img src="assets\update.png"/>

### 1.1.2 安装CUDA Toolkit

CUDA Toolkit安装包获取地址：https://developer.nvidia.com/cuda-toolkit-archive

一定要根据驱动版本来选择CUDA Toolkit，参考如下版本匹配图：

来源：https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

<img src="assets\cudatoolkit.png"/>

拿CUDA Toolkit 13.0举例，大家根据自己不同的版本去进行安装命令调试。

<img src="assets\toolkit13.0.png"/>

由于我们已经安装了显卡驱动，所以直接采用以下命令安装即可。

```python
# 下载并安装 CUDA 密钥
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 更新软件源
sudo apt update

# 安装 CUDA Toolkit（会自动安装兼容版本）
sudo apt install cuda-toolkit-13-0
```

下载安装完成后，运行以下命令查看CUDA版本

```python
nvcc -V
```

如果没有显示，可以运行如下命令后再次验证安装。

```python
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

输出如下证明安装成功：

<img src="assets\nvcc.png"/>

### 1.1.3 CUDNN安装

CUDNN安装包获取地址：https://developer.nvidia.com/cudnn-archive

CUDNN安装需要仔细查看显卡驱动和CUDA Toolkit的版本，给大家列出来查看。

也可自行前往链接查看：https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html#support-matrix

<img src="assets\cudnn.png"/>

### 1.1.4 vscode安装

vscode安装包获取地址：https://code.visualstudio.com/Download

<img src="assets\vscode.png"/>

下载linux版本的deb文件，并运行以下命令，将后面的文件名换成下载的名称：

```python
sudo dpkg -i yourfile.deb
```

### 1.1.5 miniconda安装

当然你也可以下载anaconda，跟miniconda一样使用，miniconda更加的轻量化。

miniconda安装包获取地址：https://repo.anaconda.com/miniconda/

选择Miniconda3-latest-Linux-x86_64.sh，当然也可以选择其他的安装包，根据自己需要选择

赋予执行权限：

```python
chmod +x Miniconda3-latest-Linux-x86_64.sh
```

执行安装脚本：

```python
./Miniconda3-latest-Linux-x86_64.sh
```

同时打开.bashrc文件，在底部添加如下内容（本地电脑按ctrl+h，即可打开隐藏路径，通常这个文件在home里）

```python
export PATH=$HOME/miniconda3/bin:$PATH
source ~/.bashrc
```

验证安装：

```python
conda --version
```

conda常用命令：

* 创建环境

```python
conda create -n myenv python=3.10 （创建名为 myenv 的 Python 3.10 环境）
```

* 环境查询

```python
conda env list
```

* 激活环境

```python
conda activate myenv
```

* 安装包

```python
conda install numpy
```

* 退出环境

```python
conda deactivate myenv
```

### 1.1.6 ros2humble安装

使用鱼香ROS一键安装脚本即可：

```python
wget http://fishros.com/install -O fishros && . fishros
```

## 1.2 Isaacsim 5.1安装

Isaacsim 5.1的安装包获取地址：https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html

<img src="assets\isaacsim5.1.png"/>

将这些包下载下来，第一个是isaacsim的原生软件，Isaac Sim Assets为后续可能会用到的资产。

下载完毕后，将Isaacsim以及Isaac Sim Assets压缩包移到home目录下并解压，即可运行isaac-sim。

```python
cd isaac-sim
./isaac-sim.selector.sh
```

## 1.3 IsaacLab 2.3.1安装

首先获取IsaacLab的官方源码：

```python
git clone git@github.com:isaac-sim/IsaacLab.git
#运行不了就用下面这个命令
git clone https://github.com/isaac-sim/IsaacLab.git
```

进入仓库，并设置软连接

```python
cd IsaacLab
# 请将 ${ISAACSIM_PATH} 替换为你实际的 Isaac Sim 安装路径
# 例如：ln -s ${HOME}/isaacsim _isaac_sim
ln -s ${ISAACSIM_PATH} _isaac_sim
```

创建虚拟环境并激活

```python
# 创建环境
./isaaclab.sh --conda

# 激活环境
conda activate env_isaaclab
```

安装一些必要的编译工具：

```python
sudo apt install cmake build-essential
```

运行安装命令，这会遍历源码目录中的所有扩展并使用pip进行可编辑安装

```python
./isaaclab.sh --install
```

运行以下命令，如果弹出如下窗口，说明安装成功：

```python
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

如果出现测试框架扩展模块错误，可以忽略，不影响核心功能：

```python
AttributeError: module 'omni.kit' has no attribute 'test'
```

现在就可以训练你的机器狗啦~

```python
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

示例输出如下：

```python
################################################################################
                       Learning iteration 1/1500                        

                       Computation: 58702 steps/s (collection: 1.622s, learning 0.053s)
             Mean action noise std: 0.99
          Mean value_function loss: 0.0373
               Mean surrogate loss: -0.0023
                 Mean entropy loss: 16.9891
                       Mean reward: -1.34
               Mean episode length: 40.24
Episode_Reward/track_lin_vel_xy_exp: 0.0063
Episode_Reward/track_ang_vel_z_exp: 0.0053
       Episode_Reward/lin_vel_z_l2: -0.0232
      Episode_Reward/ang_vel_xy_l2: -0.0096
     Episode_Reward/dof_torques_l2: -0.0047
         Episode_Reward/dof_acc_l2: -0.0207
     Episode_Reward/action_rate_l2: -0.0086
      Episode_Reward/feet_air_time: -0.0009
 Episode_Reward/undesired_contacts: -0.0087
Episode_Reward/flat_orientation_l2: 0.0000
     Episode_Reward/dof_pos_limits: 0.0000
         Curriculum/terrain_levels: 3.4962
Metrics/base_velocity/error_vel_xy: 0.0631
Metrics/base_velocity/error_vel_yaw: 0.0577
      Episode_Termination/time_out: 0.0331
  Episode_Termination/base_contact: 0.0103
--------------------------------------------------------------------------------
                   Total timesteps: 196608
                    Iteration time: 1.67s
                      Time elapsed: 00:00:04
                               ETA: 00:52:37
```

