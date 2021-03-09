from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path


if __name__ == "__main__":

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])
    
    config_2 = import_train_configuration(config_file='training_settings_2.ini')
    config_3 = import_train_configuration(config_file='training_settings_3.ini')
    config_4 = import_train_configuration(config_file='training_settings_4.ini')
    
    #High
    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )
    
    #Low
    TrafficGen_2 = TrafficGenerator(
        config_2['max_steps'], 
        config_2['n_cars_generated']
    )
    
    #EW
    TrafficGen_3 = TrafficGenerator(
        config_3['max_steps'], 
        config_3['n_cars_generated'],
        'EW'
    )
    
    #NS
    TrafficGen_4 = TrafficGenerator(
        config_4['max_steps'], 
        config_4['n_cars_generated'],
        'NS'
    )
    

    Model = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )
    
    Model_2 = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )
    
    Model_3 = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )
    
    Model_4 = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )


    Mem = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )
    
    Memory_2 = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )
    
    Memory_3 = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )
    
    Memory_4 = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

   
    #Same visualization
    Visualization = Visualization(
        path, 
        dpi=96
    )
    
    Sim = Simulation(
        Model,
        Mem,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_cells'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs']
    )
    
    Simulation_2 = Simulation(
        Model_2,
        Memory_2,
        TrafficGen_2,
        sumo_cmd,
        config_2['gamma'],
        config_2['max_steps'],
        config_2['green_duration'],
        config_2['yellow_duration'],
        config_2['num_cells'],
        config_2['num_states'],
        config_2['num_actions'],
        config_2['training_epochs']
    )
    
    Simulation_3 = Simulation(
        Model_3,
        Memory_3,
        TrafficGen_3,
        sumo_cmd,
        config_3['gamma'],
        config_3['max_steps'],
        config_3['green_duration'],
        config_3['yellow_duration'],
        config_3['num_cells'],
        config_3['num_states'],
        config_3['num_actions'],
        config_3['training_epochs']
    )
       
    Simulation_4 = Simulation(
        Model_4,
        Memory_4,
        TrafficGen_4,
        sumo_cmd,
        config_4['gamma'],
        config_4['max_steps'],
        config_4['green_duration'],
        config_4['yellow_duration'],
        config_4['num_cells'],
        config_4['num_states'],
        config_4['num_actions'],
        config_4['training_epochs']
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    while episode < config['total_episodes']:
        print("\n\nHigh-traffic scenario")
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = Sim.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        
        print("\n\nLow-traffic scenario")
        simulation_time, training_time = Simulation_2.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        
        print("\n\nEW-traffic scenario")
        simulation_time, training_time = Simulation_3.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        
        
        print("\n\nNS-traffic scenario")
        simulation_time, training_time = Simulation_4.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        
        
        
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    Model.save_model(path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    #Visualization.save_data_and_plot(data=Sim.reward_store, filename='reward_low', xlabel='Episode', ylabel='Cumulative negative reward')
    #Visualization.save_data_and_plot(data=Sim.cumulative_wait_store, filename='delay_low', xlabel='Episode', ylabel='Cumulative delay (s)')
    #Visualization.save_data_and_plot(data=Sim.avg_queue_length_store, filename='queue_low', xlabel='Episode', ylabel='Average queue length (vehicles)')

    
    Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.reward_store, Simulation_2.reward_store, Simulation_3.reward_store, Simulation_4.reward_store], filename='reward', xlabel='Episode', ylabel='Cumulative negative reward', scenarios=['High', 'Low', 'EW', 'NS'])
    Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.cumulative_wait_store, Simulation_2.cumulative_wait_store, Simulation_3.cumulative_wait_store, Simulation_4.cumulative_wait_store], filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)', scenarios=['High', 'Low', 'EW', 'NS'])
    Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.avg_queue_length_store, Simulation_2.avg_queue_length_store,  Simulation_3.avg_queue_length_store,  Simulation_4.avg_queue_length_store], filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)', scenarios=['High', 'low', 'EW', 'NS'])
