# Deep Q-Learning Multi Agent Perspective for Traffic Signal Control

This repository is forked from [Andrea Vidali's repository](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control) which is a framework where a deep Q-Learning Reinforcement Learning agent tries to choose the correct traffic light phase at an intersection to maximize traffic efficiency.
[Here](http://ceur-ws.org/Vol-2404/paper07.pdf) is the workshop paper published. 

## Brief description of the scope and objectives of the thesis:

The thesis is aimed at, first, studying the experimental setup and baseline approach of an adaptive traffic signal control system (TLSC) while adding some improvements.

The project goal would, then, be the passage to a multi-agent perspective by implementing two agents independently learning within the traffic environment and analysing their behaviors.

## Getting started

The steps you have to follow are perfectly described [here](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control#getting-started).

However if you have a hard time running the code on your GPU (version compatibility is the worst thing)
Take a look at [this table](https://www.tensorflow.org/install/source_windows#gpu) from the official tensorflow documentation. You must have the right versions of `CUDA` and `cuDNN` corresponding to the ones of `tensorflow-gpu` and `python`.

Here is the list of the conda package I use to run on an NVIDIA RTX 2070 Max-Q :
```
python : 3.8.8
tensorflow-gpu : 2.4.0
cudatoolkit : 11.0
cudnn : 8.1.1
```

> Note : it is possible that anaconda does not have the latest version of `cudnn` available. If this is the case, you can download it on the [official NVIDA website](https://developer.nvidia.com/rdp/cudnn-download) and paste the dll files in `C:\Users\<name>\anaconda3\envs\<env name>\Library\bin`.


