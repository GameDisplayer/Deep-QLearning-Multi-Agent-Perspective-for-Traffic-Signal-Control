# One-Intersection simulation

This section is based on [this section](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control#running-the-algorithm).

## Running the algorithm

1. Clone or download the repo.
2. Launch the script train.sh
     ```shell
    ./train.sh
    ```
    OR 
    run the following files (in order !) by executing:
    ```python
    python web_agent_training.py
    ```
    ```python
    python training_main.py
    ```
Now the agent should start the training.

You don't need to open any SUMO software since everything is loaded and done in the background. If you want to see the training process as it goes, you need to set to True the parameter gui contained in the file training_settings.ini. Keep in mind that viewing the simulation is very slow compared to the background training, and you also need to close SUMO-GUI every time an episode ends, which is not practical.

The file training_settings.ini contains all the different parameters used by the agent in the simulation.

When the training ends, the results will be stored in "./model/model_x/" where x is an increasing integer starting from 1, generated automatically. Results will include some graphs, the data used to create the graphs, the trained neural network, and a copy of the ini file where the agent settings are.

Now you can finally test the trained agent. To do so, you have to run the file testing_main.py. The test involves 5 episodes of simulation (with different seeds), and the results of the test will be stored in "./model/model_x/test/" where x is the number of the model that you specified to test. The number of the model to test and other useful parameters are contained in the file testing_settings.ini.