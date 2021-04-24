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


def avg_occupancy_and_flow(list_occupancy, list_flow):
        avg_occ = [sum(i)/len(list_occupancy) for i in zip(*list_occupancy)]
        o_max = max(avg_occ) #maximum occupancy
        max_index = avg_occ.index(o_max)
        avg_occ = avg_occ[:max_index+1]
        avg_flow = [sum(i)/len(list_flow) for i in zip(*list_flow)][:max_index+1]
        return avg_occ, avg_flow

def avg_density_and_flow(list_density, list_flow):
     avg_den = [sum(i)/len(list_density) for i in zip(*list_density)]
     d_max = max(avg_den) #maximum density
     max_index = avg_den.index(d_max)
     avg_density = avg_den[:max_index+1]
     avg_flow = [sum(i)/len(list_flow) for i in zip(*list_flow)][:max_index+1]
     return avg_density, avg_flow

def gpu_available():
    if tf.test.gpu_device_name(): 
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
        
        
def launch_process(simulation, episode, epsilon, mode, return_dict):
    simulation.run(episode, epsilon)
    return_dict[mode] = simulation.stop()
        

if __name__ == "__main__":
    
    
    #does your GPU is available ?
    gpu_available()
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    
    print("Number of processors: ", mp.cpu_count())
    
    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])
    
    
    #First idea : test an important traffic EW
    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated'],
        'EW'
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

    #inititalization of agent
    print("Initialization of the agent")
    requests.post('http://127.0.0.1:5000/initialize_agents', json={'num_layers': config['num_layers'], 
        'width_layers': config['width_layers'], 
        'batch_size': config['batch_size'], 
        'learning_rate': config['learning_rate'], 
        'num_states': config['num_states'], 
        'num_actions': config['num_actions'],
        'memory_size_max': config['memory_size_max'], 
        'memory_size_min': config['memory_size_min']})
    #Statistics
    REWARD_STORE = [] #Global negative reward
    REWARD_STORE_A1 = [] #Local negative reward first agent
    REWARD_STORE_A2 = [] #Local negative reward second agent
    CUMULATIVE_WAIT_STORE = [] #Global cumulative wait store
    CUMULATIVE_WAIT_STORE_A1 = [] #Local cumulative wait store first agent
    CUMULATIVE_WAIT_STORE_A2 = [] #Local cumulative wait store second agent
    AVG_QUEUE_LENGTH_STORE = [] #Global average queue length store
    AVG_QUEUE_LENGTH_STORE_A1 = [] #Local average queue length store first agent
    AVG_QUEUE_LENGTH_STORE_A2 = [] #Local average queue length store second agent
    AVG_WAIT_TIME_PER_VEHICLE = [] #Global average time per vehicle
    AVG_WAIT_TIME_PER_VEHICLE_A1 = [] #Local average time per vehicle first agent
    AVG_WAIT_TIME_PER_VEHICLE_A2 = [] #Local average time per vehicle second agent
    MIN_LOSS_A1 = []
    AVG_LOSS_A1 = []
    MIN_LOSS_A2 = []
    AVG_LOSS_A2 = []
    DENSITY = []
    FLOW = []
    OCCUPANCY = []
    

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
        sims=[Sim]
        mode=['EW']
        for i in range(len(sims)):
            pool.apply(launch_process, (sims[i], episode, epsilon, mode[i], return_dict),)
        pool.close()
        pool.join()
        simulation_time = round(timeit.default_timer() - start_sim_time, 1)
        print('Simulation time: ', simulation_time)
        
        #Replay
        print("Training...")
        start_time = timeit.default_timer()
        print("First agent replays")
        model_loss_agent_one=[]
        num_agent = 1
        for _ in range(config['training_epochs']):
            tr_loss = requests.post('http://127.0.0.1:5000/replay', json={'num_states': config['num_states'],
                                                              'num_actions': config['num_actions'],
                                                              'gamma': config['gamma'],
                                                              'num_agent': num_agent}).json()['loss']
            model_loss_agent_one.append(tr_loss)
            
        print("Second agent replays")
        model_loss_agent_two=[]
        num_agent = 2
        for _ in range(config['training_epochs']):
            tr_loss = requests.post('http://127.0.0.1:5000/replay', json={'num_states': config['num_states'],
                                                              'num_actions': config['num_actions'],
                                                              'gamma': config['gamma'],
                                                              'num_agent': num_agent}).json()['loss']
            model_loss_agent_two.append(tr_loss)
            
        training_time = round(timeit.default_timer() - start_time, 1)
        print('Training time: ', training_time)
        
        print('\nTotal time for this simulation: ', simulation_time+training_time)
        
        print("Saving loss results...")
        if(len(model_loss_agent_one) > 0):
             AVG_LOSS_A1.append(sum(model_loss_agent_one)/config['training_epochs'])
             MIN_LOSS_A1.append(min(model_loss_agent_one))
             
        if(len(model_loss_agent_two) > 0):
             AVG_LOSS_A2.append(sum(model_loss_agent_two)/config['training_epochs'])
             MIN_LOSS_A2.append(min(model_loss_agent_two))
        
            
          
        for m in mode:
            REWARD_STORE.append(return_dict[m][0])
            REWARD_STORE_A1.append(return_dict[m][1])
            REWARD_STORE_A2.append(return_dict[m][2])
            CUMULATIVE_WAIT_STORE.append(return_dict[m][3])
            CUMULATIVE_WAIT_STORE_A1.append(return_dict[m][4])
            CUMULATIVE_WAIT_STORE_A2.append(return_dict[m][5])
            AVG_QUEUE_LENGTH_STORE.append(return_dict[m][6])
            AVG_QUEUE_LENGTH_STORE_A1.append(return_dict[m][7])
            AVG_QUEUE_LENGTH_STORE_A2.append(return_dict[m][8])
            AVG_WAIT_TIME_PER_VEHICLE.append(return_dict[m][9])
            AVG_WAIT_TIME_PER_VEHICLE_A1.append(return_dict[m][10])
            AVG_WAIT_TIME_PER_VEHICLE_A2.append(return_dict[m][11])
            DENSITY.append(return_dict[m][12])
            FLOW.append(return_dict[m][13])
            OCCUPANCY.append(return_dict[m][14])
            
        
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    requests.post('http://127.0.0.1:5000/save_models', json={'path': path})
    #Model.save_model(path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))
    
    
    print("\nPlotting the aggregate global measures...")
    Visualization.save_data_and_plot(data=REWARD_STORE, filename='negative_reward', title="Cumulative negative reward per episode", xlabel='Episodes', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=CUMULATIVE_WAIT_STORE, filename='cum_delay', title="Cumulative delay per episode", xlabel='Episodes', ylabel='Cumulative delay [s]')
    Visualization.save_data_and_plot(data=AVG_QUEUE_LENGTH_STORE, filename='queue',title="Average queue length per episode", xlabel='Episodes', ylabel='Average queue length [vehicles]')
    Visualization.save_data_and_plot(data=AVG_WAIT_TIME_PER_VEHICLE, filename='wait_per_vehicle', title="Average waiting time per vehicle per episode", xlabel='Episodes', ylabel='Average waiting time per vehicle [s]')
    
    print("\nPlotting the aggregate local measures for agent 1 (left intersection)...")
    Visualization.save_data_and_plot(data=REWARD_STORE_A1, filename='negative_reward_agent_one', title="Cumulative negative reward per episode", xlabel='Episodes', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=CUMULATIVE_WAIT_STORE_A1, filename='cum_delay_agent_one', title="Cumulative delay per episode", xlabel='Episodes', ylabel='Cumulative delay [s]')
    Visualization.save_data_and_plot(data=AVG_QUEUE_LENGTH_STORE_A1, filename='queue_agent_one',title="Average queue length per episode", xlabel='Episodes', ylabel='Average queue length [vehicles]')
    Visualization.save_data_and_plot(data=AVG_WAIT_TIME_PER_VEHICLE_A1, filename='wait_per_vehicle_agent_one', title="Average waiting time per vehicle per episode", xlabel='Episodes', ylabel='Average waiting time per vehicle [s]')
    
    
    print("\nPlotting the aggregate local measures for agent 2 (right intersection)...")
    Visualization.save_data_and_plot(data=REWARD_STORE_A2, filename='negative_reward_agent_two', title="Cumulative negative reward per episode", xlabel='Episodes', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=CUMULATIVE_WAIT_STORE_A2, filename='cum_delay_agent_two', title="Cumulative delay per episode", xlabel='Episodes', ylabel='Cumulative delay [s]')
    Visualization.save_data_and_plot(data=AVG_QUEUE_LENGTH_STORE_A2, filename='queue_agent_two',title="Average queue length per episode", xlabel='Episodes', ylabel='Average queue length [vehicles]')
    Visualization.save_data_and_plot(data=AVG_WAIT_TIME_PER_VEHICLE_A2, filename='wait_per_vehicle_agent_two', title="Average waiting time per vehicle per episode", xlabel='Episodes', ylabel='Average waiting time per vehicle [s]')
    
    
    print("\nCalculating Average loss of models...")
    Visualization.save_data_and_plot(data=MIN_LOSS_A1, filename='min_loss_agent_one', title="Minimum MAE loss of the first model per episode", xlabel='Episodes', ylabel='Minimum MAE')
    Visualization.save_data_and_plot(data=AVG_LOSS_A1, filename='avg_loss_agent_one', title="Average MAE loss of the first model per episode", xlabel='Episodes', ylabel='Average MAE')

    Visualization.save_data_and_plot(data=MIN_LOSS_A2, filename='min_loss_agent_two', title="Minimum MAE loss of the second model per episode", xlabel='Episodes', ylabel='Minimum MAE')
    Visualization.save_data_and_plot(data=AVG_LOSS_A2, filename='avg_loss_agent_two', title="Average MAE loss of the second model per episode", xlabel='Episodes', ylabel='Average MAE')


    # print("\nPlotting the fundamental diagrams of traffic flow depending on the scenario...")
    # s1 = avg_density_and_flow([DENSITY[i] for i in range(len(DENSITY)) if i%4==0]  , [FLOW[i] for i in range(len(FLOW)) if i%4==0])
    # s2 = avg_density_and_flow([DENSITY[i] for i in range(len(DENSITY)) if i%4==1]  , [FLOW[i] for i in range(len(FLOW)) if i%4==1])
    # s3 = avg_density_and_flow([DENSITY[i] for i in range(len(DENSITY)) if i%4==2]  , [FLOW[i] for i in range(len(FLOW)) if i%4==2])
    # s4 = avg_density_and_flow([DENSITY[i] for i in range(len(DENSITY)) if i%4==3]  , [FLOW[i] for i in range(len(FLOW)) if i%4==3])

    # Visualization.save_data_and_plot_fundamental_diagram(data=s1, filename='fundamental_diagram_High', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='High')
    # Visualization.save_data_and_plot_fundamental_diagram(data=s2, filename='fundamental_diagram_Low', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='Low')
    # Visualization.save_data_and_plot_fundamental_diagram(data=s3, filename='fundamental_diagram_EW', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='EW')
    # Visualization.save_data_and_plot_fundamental_diagram(data=s4, filename='fundamental_diagram_NS', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenario='NS')

    # Visualization.save_data_and_plot_multiple_fundamental_diagram(data=[s1, s2, s3, s4], filename='fundamental_diagram', xlabel='Density [vehicles per km]', ylabel='Flow [vehicles per hour]', scenarios=['High', 'Low', 'EW', 'NS'])

    # print("\nPlotting the occupancy diagrams of traffic flow depending on the scenario...")
    
    # o1 = avg_occupancy_and_flow([OCCUPANCY[i] for i in range(len(OCCUPANCY)) if i%4==0]  , [FLOW[i] for i in range(len(FLOW)) if i%4==0])
    # o2 = avg_occupancy_and_flow([OCCUPANCY[i] for i in range(len(OCCUPANCY)) if i%4==1]  , [FLOW[i] for i in range(len(FLOW)) if i%4==1])
    # o3 = avg_occupancy_and_flow([OCCUPANCY[i] for i in range(len(OCCUPANCY)) if i%4==2]  , [FLOW[i] for i in range(len(FLOW)) if i%4==2])
    # o4 = avg_occupancy_and_flow([OCCUPANCY[i] for i in range(len(OCCUPANCY)) if i%4==3]  , [FLOW[i] for i in range(len(FLOW)) if i%4==3])

    # Visualization.save_data_and_plot_fundamental_diagram(data=o1, filename='occ_fundamental_diagram_High', xlabel='Occupancy [%]', ylabel='Flow [vehicles per hour]', scenario='High')
    # Visualization.save_data_and_plot_fundamental_diagram(data=o2, filename='occ_fundamental_diagram_Low', xlabel='Occupancy [%]', ylabel='Flow [vehicles per hour]', scenario='Low')
    # Visualization.save_data_and_plot_fundamental_diagram(data=o3, filename='occ_fundamental_diagram_EW', xlabel='Occupancy [%]', ylabel='Flow [vehicles per hour]', scenario='EW')
    # Visualization.save_data_and_plot_fundamental_diagram(data=o4, filename='occ_fundamental_diagram_NS', xlabel='Occupancy [%]', ylabel='Flow [vehicles per hour]', scenario='NS')

    # Visualization.save_data_and_plot_multiple_fundamental_diagram(data=[o1, o2, o3, o4], filename='occ_fundamental_diagram', xlabel='Occupancy [%]', ylabel='Flow [vehicles per hour]', scenarios=['High', 'Low', 'EW', 'NS'])

