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
    
    #High
    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated_high']
    )
    
    #Low
    TrafficGen_2 = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated_low']
    )
    
    #EW
    TrafficGen_3 = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated_ew'],
        'EW'
    )
    
    #NS
    TrafficGen_4 = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated_ns'],
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
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_cells'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs']
    )
    
    Simulation_3 = Simulation(
        Model_3,
        Memory_3,
        TrafficGen_3,
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
       
    Simulation_4 = Simulation(
        Model_4,
        Memory_4,
        TrafficGen_4,
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

    Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.reward_store, Simulation_2.reward_store, Simulation_3.reward_store, Simulation_4.reward_store], filename='negative_reward', title="Cumulative negative reward per episode", xlabel='Episodes', ylabel='Cumulative negative reward', scenarios=['High', 'Low', 'EW', 'NS'])
    Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.cumulative_wait_store, Simulation_2.cumulative_wait_store, Simulation_3.cumulative_wait_store, Simulation_4.cumulative_wait_store], filename='cum_delay', title="Cumulative delay per episode", xlabel='Episodes', ylabel='Cumulative delay [s]', scenarios=['High', 'Low', 'EW', 'NS'])
    Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.avg_queue_length_store, Simulation_2.avg_queue_length_store,  Simulation_3.avg_queue_length_store,  Simulation_4.avg_queue_length_store], filename='queue',title="Average queue length per episode", xlabel='Episodes', ylabel='Average queue length [vehicles]', scenarios=['High', 'Low', 'EW', 'NS'])
    Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.avg_wait_time_per_vehicle, Simulation_2.avg_wait_time_per_vehicle,  Simulation_3.avg_wait_time_per_vehicle,  Simulation_4.avg_wait_time_per_vehicle], filename='wait_per_vehicle', title="Average waiting time per vehicle per episode", xlabel='Episodes', ylabel='Average waiting time per vehicle [s]', scenarios=['High', 'Low', 'EW', 'NS'])
    Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.min_loss, Simulation_2.min_loss,  Simulation_3.min_loss,  Simulation_4.min_loss], filename='min_loss', title="Minimum MAE loss of the model per episode", xlabel='Episodes', ylabel='Minimum MAE', scenarios=['High', 'Low', 'EW', 'NS'])
    print("\nPlotting the fundamental traffic flow...")
    Visualization.save_data_and_plot_fundamental_diagram(density_and_flow=[Sim.avg_density_and_flow, Simulation_2.avg_density_and_flow, Simulation_3.avg_density_and_flow, Simulation_4.avg_density_and_flow], filename='fundamental_diagram', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenarios=['High', 'Low', 'EW', 'NS'])
    print("\nCalculating Average loss of model...")
    Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.avg_loss, Simulation_2.avg_loss, Simulation_3.avg_loss, Simulation_4.avg_loss], filename='loss', title="Average MAE loss of the model per episode", xlabel='Episodes', ylabel='Average MAE', scenarios=['High', 'Low', 'EW', 'NS'])
