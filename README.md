# Deep Q-Learning Multi Agent Perspective for Traffic Signal Control

This repository is forked from [Andrea Vidali's repository](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control) which is a framework where a deep Q-Learning Reinforcement Learning agent tries to choose the correct traffic light phase at an intersection to maximize traffic efficiency.
[Here](http://ceur-ws.org/Vol-2404/paper07.pdf) is the workshop paper published. 

## Brief description of the scope and objectives of the thesis:

The thesis is aimed at, first, studying the experimental setup and baseline approach of an adaptive traffic signal control system (TLSC) while adding some improvements.

The project goal would, then, be the passage to a multi-agent perspective by implementing two agents independently learning within the traffic environment and analysing their behaviors.

## Getting started

The first steps you have to follow are perfectly described [here](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control#getting-started).

However if you have a hard time running the code on your GPU (version compatibility is the worst thing)
Take a look at [this table](https://www.tensorflow.org/install/source_windows#gpu) from the official tensorflow documentation. You must have the right versions of `CUDA` and `cuDNN` corresponding to the ones of `tensorflow-gpu` and `python`.

Here is the list of the conda package I use to run on an NVIDIA RTX 2070 Max-Q :
```
python : 3.8.8
tensorflow-gpu : 2.4.0
cudatoolkit : 11.0
cudnn : 8.1.1
```

> Note : it is possible that anaconda does not have the latest version of `cudnn` available. If this is the case, you can download it on the [official NVIDIA website](https://developer.nvidia.com/rdp/cudnn-download) and paste the dll files in `C:\Users\<name>\anaconda3\envs\<env name>\Library\bin`.

## Code structure :

- `TLCS` - combines the scripts for the 1-only intersection [readme in construction](https://github.com/GameDisplayer/Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control/tree/master/TLCS#readme)
- `2TLCS` - focus on the 2-intersection simulations [readme in construction](https://github.com/GameDisplayer/Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control/tree/master/2TLCS#readme)
- `requirements.txt` - Output installed packages in requirements format for the Docker image explained in [this section](https://github.com/GameDisplayer/Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control#docker-hub). (*See [here](https://pip.pypa.io/en/stable/cli/pip_freeze/) for more information*)


## Docker Hub
A [Docker Hub repo](https://hub.docker.com/repository/docker/gamedisplayer/sumo-experiments) has been created to allow you dockerise SUMO in order to train the agents on a choosen simulation. 


### Usage 

1. You must pull the image :

    ```shell
    docker pull gamedisplayer/sumo-experiments
    ```
2. In the main directory of the repo (at the location of the *requirements.txt*), run the container with the volume attached :

    ```shell
    docker run -it --name [the_name_you_want] -v [volume_location]:/experiment gamedisplayer/sumo-experiments /bin/bash
    ```
    Note : *[volume_location]* might correspond to something like *~/git/Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control*

4. Launch your experiments !

    Go to ```TLCS``` or ```2TLCS``` and run the bash script :
    ```shell
    ./train.sh
    ```

For more information (environment variables or description) please go to the [Docker Hub repo](https://hub.docker.com/repository/docker/gamedisplayer/sumo-experiments) !

