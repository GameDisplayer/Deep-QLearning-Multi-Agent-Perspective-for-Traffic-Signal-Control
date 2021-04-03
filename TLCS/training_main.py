from __future__ import absolute_import
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
from shutil import copyfile

from training_simulation import Simulation
from generator import TrafficGenerator
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path
import tensorflow as tf

import multiprocessing as mp
import requests

import timeit


def avg_density_and_flow(list_density, list_flow):
     max_steps=5400
     avg_den = [sum(i)/max_steps for i in zip(*list_density)]
     d_max = max(avg_den) #maximum density
     max_index = avg_den.index(d_max)
     avg_density = avg_den[:max_index+1]
     avg_flow = [sum(i)/max_steps for i in zip(*list_flow)][:max_index+1]
     return avg_density, avg_flow

def gpu_available():
    if tf.test.gpu_device_name(): 
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
        
        
def launch_process(simulation, episode, epsilon, mode, return_dict):
    simulation.run(episode, epsilon)
    print()
    return_dict[mode] = simulation.stop()
        

if __name__ == "__main__":
    
    
    #does your GPU is available ?
    gpu_available()
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    
    print("Number of processors: ", mp.cpu_count())
    
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
   
    #Same visualization
    Visualization = Visualization(
        path, 
        dpi=96
    )
    
    Sim = Simulation(
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
    

    #inititalization of agent
    print("Initialization of the agent")
    requests.post('http://127.0.0.1:5000/initialize_agent', json={'num_layers': config['num_layers'], 
        'width_layers': config['width_layers'], 
        'batch_size': config['batch_size'], 
        'learning_rate': config['learning_rate'], 
        'num_states': config['num_states'], 
        'num_actions': config['num_actions'],
        'memory_size_max': config['memory_size_max'], 
        'memory_size_min': config['memory_size_min']})
    
    #Statistics
    REWARD_STORE = []
    CUMULATIVE_WAIT_STORE = []
    AVG_QUEUE_LENGTH_STORE = []
    AVG_WAIT_TIME_PER_VEHICLE = []
    MIN_LOSS = []
    AVG_LOSS = []
    DENSITY = []
    FLOW = []
    

    episode = 0
    timestamp_start = datetime.datetime.now()
    while episode < config['total_episodes']:
        
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
    
        manager = mp.Manager()
        return_dict = manager.dict()
        
        print("Launch processes")
        start_sim_time = timeit.default_timer()
        pool = mp.Pool(processes=12)
        sims=[Sim, Simulation_2, Simulation_3, Simulation_4]
        mode=['HIGH', 'LOW', 'EW', 'NS']
        for i in range(len(sims)):
            pool.apply(launch_process, (sims[i], episode, epsilon, mode[i], return_dict),)
        pool.close()
        pool.join()
        simulation_time = round(timeit.default_timer() - start_sim_time, 1)
        print('Simulation time:', simulation_time)
        
        #Replay
        print("Training...")
        start_time = timeit.default_timer()
        model_loss=[]
        for _ in range(config['training_epochs']):
            tr_loss = requests.post('http://127.0.0.1:5000/replay', json={'num_states': config['num_states'],
                                                              'num_actions': config['num_actions'],
                                                              'gamma': config['gamma']}).json()['loss']
            model_loss.append(tr_loss)
        training_time = round(timeit.default_timer() - start_time, 1)
        print('Training time :', training_time)
        
        if(len(model_loss) > 0):
             print("Saving loss results...")
             #print(self._model_training_loss)
             AVG_LOSS.append(sum(model_loss)/config['training_epochs'])
             MIN_LOSS.append(min(model_loss))
            
        
        
        
        for m in mode:
            REWARD_STORE.append(return_dict[m][0])
            CUMULATIVE_WAIT_STORE.append(return_dict[m][1])
            AVG_QUEUE_LENGTH_STORE.append(return_dict[m][2])
            AVG_WAIT_TIME_PER_VEHICLE.append(return_dict[m][3])
            #MIN_LOSS.append(return_dict[m][4])
            #AVG_LOSS.append(return_dict[m][5])
            DENSITY.append(return_dict[m][4])
            FLOW.append(return_dict[m][5])
            
        
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    requests.post('http://127.0.0.1:5000/save_model', json={'path': path})
    #Model.save_model(path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))
    
    Visualization.save_data_and_plot_multiple_curves(list_of_data=[[REWARD_STORE[i] for i in range(len(REWARD_STORE)) if i%4==0], [REWARD_STORE[i] for i in range(len(REWARD_STORE)) if i%4==1], [REWARD_STORE[i] for i in range(len(REWARD_STORE)) if i%4==2], [REWARD_STORE[i] for i in range(len(REWARD_STORE)) if i%4==3]], filename='negative_reward', title="Cumulative negative reward per episode", xlabel='Episodes', ylabel='Cumulative negative reward', scenarios=['High', 'Low', 'EW', 'NS'])
    Visualization.save_data_and_plot_multiple_curves(list_of_data=[[CUMULATIVE_WAIT_STORE[i] for i in range(len(CUMULATIVE_WAIT_STORE)) if i%4==0], [CUMULATIVE_WAIT_STORE[i] for i in range(len(CUMULATIVE_WAIT_STORE)) if i%4==1], [CUMULATIVE_WAIT_STORE[i] for i in range(len(CUMULATIVE_WAIT_STORE)) if i%4==2], [CUMULATIVE_WAIT_STORE[i] for i in range(len(CUMULATIVE_WAIT_STORE)) if i%4==3]], filename='cum_delay', title="Cumulative delay per episode", xlabel='Episodes', ylabel='Cumulative delay [s]', scenarios=['High', 'Low', 'EW', 'NS'])
    Visualization.save_data_and_plot_multiple_curves(list_of_data=[[AVG_QUEUE_LENGTH_STORE[i] for i in range(len(AVG_QUEUE_LENGTH_STORE)) if i%4==0], [AVG_QUEUE_LENGTH_STORE[i] for i in range(len(AVG_QUEUE_LENGTH_STORE)) if i%4==1],  [AVG_QUEUE_LENGTH_STORE[i] for i in range(len(AVG_QUEUE_LENGTH_STORE)) if i%4==2],  [AVG_QUEUE_LENGTH_STORE[i] for i in range(len(AVG_QUEUE_LENGTH_STORE)) if i%4==3]], filename='queue',title="Average queue length per episode", xlabel='Episodes', ylabel='Average queue length [vehicles]', scenarios=['High', 'Low', 'EW', 'NS'])
    Visualization.save_data_and_plot_multiple_curves(list_of_data=[[AVG_WAIT_TIME_PER_VEHICLE[i] for i in range(len(AVG_WAIT_TIME_PER_VEHICLE)) if i%4==0], [AVG_WAIT_TIME_PER_VEHICLE[i] for i in range(len(AVG_WAIT_TIME_PER_VEHICLE)) if i%4==1],  [AVG_WAIT_TIME_PER_VEHICLE[i] for i in range(len(AVG_WAIT_TIME_PER_VEHICLE)) if i%4==2],  [AVG_WAIT_TIME_PER_VEHICLE[i] for i in range(len(AVG_WAIT_TIME_PER_VEHICLE)) if i%4==3]], filename='wait_per_vehicle', title="Average waiting time per vehicle per episode", xlabel='Episodes', ylabel='Average waiting time per vehicle [s]', scenarios=['High', 'Low', 'EW', 'NS'])
    #Visualization.save_data_and_plot_multiple_curves(list_of_data=[[MIN_LOSS[i] for i in range(len(MIN_LOSS)) if i%4==0], [MIN_LOSS[i] for i in range(len(MIN_LOSS)) if i%4==1],  [MIN_LOSS[i] for i in range(len(MIN_LOSS)) if i%4==2],  [MIN_LOSS[i] for i in range(len(MIN_LOSS)) if i%4==3]], filename='min_loss', title="Minimum MAE loss of the model per episode", xlabel='Episodes', ylabel='Minimum MAE', scenarios=['High', 'Low', 'EW', 'NS'])
    print("\nCalculating Average loss of model...")
    #Visualization.save_data_and_plot_multiple_curves(list_of_data=[[AVG_LOSS[i] for i in range(len(AVG_LOSS)) if i%4==0], [AVG_LOSS[i] for i in range(len(AVG_LOSS)) if i%4==1], [AVG_LOSS[i] for i in range(len(AVG_LOSS)) if i%4==2], [AVG_LOSS[i] for i in range(len(AVG_LOSS)) if i%4==3]], filename='loss', title="Average MAE loss of the model per episode", xlabel='Episodes', ylabel='Average MAE', scenarios=['High', 'Low', 'EW', 'NS'])
    Visualization.save_data_and_plot(data=MIN_LOSS, filename='min_loss', title="Minimum MAE loss of the model per episode", xlabel='Episodes', ylabel='Minimum MAE')
    Visualization.save_data_and_plot(data=AVG_LOSS, filename='avg_loss', title="Average MAE loss of the model per episode", xlabel='Episodes', ylabel='Average MAE')

    s1 = avg_density_and_flow([DENSITY[i] for i in range(len(DENSITY)) if i%4==0]  , [FLOW[i] for i in range(len(FLOW)) if i%4==0])
    s2 = avg_density_and_flow([DENSITY[i] for i in range(len(DENSITY)) if i%4==1]  , [FLOW[i] for i in range(len(FLOW)) if i%4==1])
    s3 = avg_density_and_flow([DENSITY[i] for i in range(len(DENSITY)) if i%4==2]  , [FLOW[i] for i in range(len(FLOW)) if i%4==2])
    s4 = avg_density_and_flow([DENSITY[i] for i in range(len(DENSITY)) if i%4==3]  , [FLOW[i] for i in range(len(FLOW)) if i%4==3])


    print("\nPlotting the fundamental diagrams of traffic flow depending on the scenario...")
    Visualization.save_data_and_plot_fundamental_diagram(density_and_flow=s1, filename='fundamental_diagram_High', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='High')
    Visualization.save_data_and_plot_fundamental_diagram(density_and_flow=s2, filename='fundamental_diagram_Low', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='Low')
    Visualization.save_data_and_plot_fundamental_diagram(density_and_flow=s3, filename='fundamental_diagram_EW', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='EW')
    Visualization.save_data_and_plot_fundamental_diagram(density_and_flow=s4, filename='fundamental_diagram_NS', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='NS')

    Visualization.save_data_and_plot_multiple_fundamental_diagram(density_and_flow=[s1, s2, s3, s4], filename='fundamental_diagram', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenarios=['High', 'Low', 'EW', 'NS'])

