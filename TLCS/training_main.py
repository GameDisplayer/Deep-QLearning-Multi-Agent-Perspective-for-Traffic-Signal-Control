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

import multiprocessing as mp
import pickle
import tensorflow as tf
from tensorflow.python.client import device_lib 


def gpu_available():
    if tf.test.gpu_device_name(): 
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
        
    

if __name__ == "__main__":
    
    #does your GPU is available ?
    gpu_available()
    
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
    

    #Agent
    Model = TrainModel(
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
   
    #Same visualization
    Visualization = Visualization(
        path, 
        dpi=96
    )
    
    Sim = Simulation(
        None,
        None,
        None,
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
        None,
        None,
        None,
        Mem,
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
    
    # Simulation_3 = Simulation(
    #     Model,
    #     Mem,
    #     TrafficGen_3,
    #     sumo_cmd,
    #     config['gamma'],
    #     config['max_steps'],
    #     config['green_duration'],
    #     config['yellow_duration'],
    #     config['num_cells'],
    #     config['num_states'],
    #     config['num_actions'],
    #     config['training_epochs']
    # )
       
    # Simulation_4 = Simulation(
    #     Model,
    #     Mem,
    #     TrafficGen_4,
    #     sumo_cmd,
    #     config['gamma'],
    #     config['max_steps'],
    #     config['green_duration'],
    #     config['yellow_duration'],
    #     config['num_cells'],
    #     config['num_states'],
    #     config['num_actions'],
    #     config['training_epochs']
    # )
    


    episode = 0
    timestamp_start = datetime.datetime.now()
    config['total_episodes'] = 5
    while episode < config['total_episodes']:
        
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        

        model = mp.JoinableQueue()
        
        print("Hey")
        Sim.epsilon = epsilon
        Sim.episode = episode
        Simulation_2.epsilon = epsilon
        Simulation_2.episode = episode
        
        
        Sim._Model = model
        Simulation_2._Model = model
        
        print("JO")
        Sim.start()
        Simulation_2.start()
        
        print("TY")
        model.put(Model)
        
        print("U")
        model.join()
        
        
      
            
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    # Model.save_model(path)

    # copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))



# =============================================================================
#     Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.reward_store, Simulation_2.reward_store, Simulation_3.reward_store, Simulation_4.reward_store], filename='negative_reward', title="Cumulative negative reward per episode", xlabel='Episodes', ylabel='Cumulative negative reward', scenarios=['High', 'Low', 'EW', 'NS'])
#     Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.cumulative_wait_store, Simulation_2.cumulative_wait_store, Simulation_3.cumulative_wait_store, Simulation_4.cumulative_wait_store], filename='cum_delay', title="Cumulative delay per episode", xlabel='Episodes', ylabel='Cumulative delay [s]', scenarios=['High', 'Low', 'EW', 'NS'])
#     Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.avg_queue_length_store, Simulation_2.avg_queue_length_store,  Simulation_3.avg_queue_length_store,  Simulation_4.avg_queue_length_store], filename='queue',title="Average queue length per episode", xlabel='Episodes', ylabel='Average queue length [vehicles]', scenarios=['High', 'Low', 'EW', 'NS'])
#     Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.avg_wait_time_per_vehicle, Simulation_2.avg_wait_time_per_vehicle,  Simulation_3.avg_wait_time_per_vehicle,  Simulation_4.avg_wait_time_per_vehicle], filename='wait_per_vehicle', title="Average waiting time per vehicle per episode", xlabel='Episodes', ylabel='Average waiting time per vehicle [s]', scenarios=['High', 'Low', 'EW', 'NS'])
#     Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.min_loss, Simulation_2.min_loss,  Simulation_3.min_loss,  Simulation_4.min_loss], filename='min_loss', title="Minimum MAE loss of the model per episode", xlabel='Episodes', ylabel='Minimum MAE', scenarios=['High', 'Low', 'EW', 'NS'])
#     print("\nPlotting the fundamental diagrams of traffic flow depending on the scenario...")
#     Visualization.save_data_and_plot_fundamental_diagram(density_and_flow=Sim.avg_density_and_flow, filename='fundamental_diagram_High', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='High')
#     Visualization.save_data_and_plot_fundamental_diagram(density_and_flow=Simulation_2.avg_density_and_flow, filename='fundamental_diagram_Low', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='Low')
#     Visualization.save_data_and_plot_fundamental_diagram(density_and_flow=Simulation_3.avg_density_and_flow, filename='fundamental_diagram_EW', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='EW')
#     Visualization.save_data_and_plot_fundamental_diagram(density_and_flow=Simulation_4.avg_density_and_flow, filename='fundamental_diagram_NS', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='NS')
# 
#     print("\nCalculating Average loss of model...")
#     Visualization.save_data_and_plot_multiple_curves(list_of_data=[Sim.avg_loss, Simulation_2.avg_loss, Simulation_3.avg_loss, Simulation_4.avg_loss], filename='loss', title="Average MAE loss of the model per episode", xlabel='Episodes', ylabel='Average MAE', scenarios=['High', 'Low', 'EW', 'NS'])
# 
#     Visualization.save_data_and_plot_multiple_fundamental_diagram(density_and_flow=[Sim.get_avg_density_and_flow, Simulation_2.get_avg_density_and_flow, Simulation_3.get_avg_density_and_flow, Simulation_4.get_avg_density_and_flow], filename='fundamental_diagram', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenarios=['High', 'Low', 'EW', 'NS'])
# =============================================================================
