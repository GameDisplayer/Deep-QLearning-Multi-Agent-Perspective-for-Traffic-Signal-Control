# Deep Q-Learning Multi Agent Perspective for Traffic Signal Control

This repository is forked from [Andrea Vidali's repository](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control) which is a framework where a deep Q-Learning Reinforcement Learning agent tries to choose the correct traffic light phase at an intersection to maximize traffic efficiency.
[Here](http://ceur-ws.org/Vol-2404/paper07.pdf) is the workshop paper published. 

## Brief description of the scope and objectives of the thesis:

The thesis is aimed at, first, studying the experimental setup and baseline approach of an adaptive traffic signal control system (TLCS) while adding some improvements.

The project goal would, then, be the passage to a multi-agent perspective by implementing two agents independently learning within the traffic environment and analysing their behaviors.

## Getting started

The first steps you have to follow are perfectly described [here](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control#getting-started).

However if you have a hard time running the code on your GPU (version compatibility is the worst thing)
Take a look at [this table](https://www.tensorflow.org/install/source_windows#gpu) from the official tensorflow documentation. You must have the right versions of `CUDA` and `cuDNN` corresponding to the ones of `tensorflow-gpu` and `python3`.

Here is the list of the conda package I use to run on an NVIDIA RTX 2070 Max-Q :
```
python : 3.8.8
tensorflow-gpu : 2.4.0
cudatoolkit : 11.0
cudnn : 8.1.1
```

> Note : it is possible that anaconda does not have the latest version of `cudnn` available. If this is the case, you can download it on the [official NVIDIA website](https://developer.nvidia.com/rdp/cudnn-download) and paste the dll files in `C:\Users\<name>\anaconda3\envs\<env name>\Library\bin`.

## Code structure

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
2. If you have already cloned the repository, just go inside the workdir:
    
    ```shell
    cd Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control
    ```
3. Check that you are in the main directory of the repo (at the location of the *requirements.txt*), then run the container with the volume attached :

    ```shell
    docker run --rm -it --name [the_name_you_want] -v $PWD:/experiment gamedisplayer/sumo-experiments /bin/bash
    ```
4. Launch your experiments !

    Go to ```TLCS``` or ```2TLCS``` and run the bash script :
    ```shell
    ./train.sh
    ```
 *Note:* If you want to quit the virtual environment, just type and enter:
    ```
    exit
    ```

For more information (environment variables or description) please go to the [Docker Hub repo](https://hub.docker.com/repository/docker/gamedisplayer/sumo-experiments) !

## Future work

During my thesis, I learned about the [actor-critic](http://incompleteideas.net/book/first/ebook/node66.html) methods that already has promising results in the Adaptive Traffic Signal Control branch.
Based on [this implementation](https://github.com/cts198859/deeprl_signal_control) I created a forked repo to make some experiments on the 2TLCS environment. It is far from being the central part of my work but I think that it deserves to be explored further. [Discover the repo](https://github.com/GameDisplayer/Multi-Agent-DRL4-Large-Scale-Traffic-Signal-Control)

## Author

Romain Michelucci - r.michelucci@campus.unimib.it

The master's thesis is fully available [here](https://github.com/GameDisplayer/Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control/tree/master/Master%20Thesis).

If you need further information about the algorithm or the report, I suggest you open an issue on the issues page or contact me to my email address. I 'll very pleased to answer it !

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/GameDisplayer/Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control/blob/master/LICENSE) file for details.
