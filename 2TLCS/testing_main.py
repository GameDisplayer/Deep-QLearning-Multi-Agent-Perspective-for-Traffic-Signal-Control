from __future__ import absolute_import
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from shutil import copyfile

from testing_simulation import Simulation
from generator import TrafficGenerator
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path

import statistics


if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated'],
        "EW"
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
        
    Simulation = Simulation(
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_cells'],
        config['num_states'],
        config['num_actions'],
        config['n_cars_generated']
    )
    
    reward=0
    episode = 0
    ql=[]
    awt=[]
    
    seed = [10000, 10001, 10002, 10003, 10004]
    while episode < 5:
        print('\n----- Test episode n°', episode)
        simulation_time = Simulation.run(seed[episode])  # run the simulation
        print('Simulation time:', simulation_time, 's')
        
        
        reward+=Simulation._sum_neg_reward_one + Simulation._sum_neg_reward_two       
        ql.append(Simulation._sum_queue_length)
        #print(sum(Simulation._waits))
        #awt.append(Simulation._sum_queue_length/sum(Simulation._waits))
        episode += 1
        
    # print('\n----- Test episode')
    # simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
    # print('Simulation time:', simulation_time, 's')

    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    #Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
    #Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue lenght (vehicles)')

    print('nrw', reward/5)
    print('twt', sum(ql)/5)
    print('awt', statistics.median(awt))


    
