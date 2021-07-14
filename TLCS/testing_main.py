from __future__ import absolute_import
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from shutil import copyfile

from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path

import statistics


if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    Model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated'],
        config['art_queue'],
        None 
    )
    #None or "EW" or "NS"

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_cells'],
        config['num_states'],
        config['num_actions'],
        config['n_cars_generated'],
        config['static_traffic_lights'] #STL or NOT
    )
    
    reward=0 #reward
    episode = 0 #episode number
    ql=[] #queue length vector for 5 episodes
    awt=[] #awt vector for 5 episodes
    
    seed = [1, 2, 3, 4, 5] #seeds for reproducibility
    while episode < 5:
        print('\n----- Test episode n°', episode)
        simulation_time = Simulation.run(seed[episode])
        print('Simulation time:', simulation_time, 's')
        
        reward+=Simulation._sum_neg_reward        
        ql.append(Simulation._sum_queue_length)
        awt.append(Simulation._sum_queue_length/sum(Simulation._waits))
        episode += 1

    print("----- Testing info saved at:", plot_path)
    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini')) #Save to recall the test settings


    #Print informations for average episodes
    print('nrw', reward/5)
    print('twt', sum(ql)/5)
    print('awt', statistics.median(awt))


    
