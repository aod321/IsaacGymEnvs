# IsaacWFC via GPU-pipeline

## Installation

可以配置Conda清华源,防止网速过慢,编辑~/.condarc文件,添加如下内容:
```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

```
之后如下执行安装即可:

```
conda create --name=issac_rlgames python=3.8
conda activate issac_rlgames
conda install pytorch cudatoolkit=11.6 -c pytorch -c conda-forge
..未完待续..
```

## Training Empty aera
```
cd isaacgymenvs/
```
```
 python train.py task=WFCIsaacTask headless=True num_envs=16 +view=False sim_device="cuda:0"
```