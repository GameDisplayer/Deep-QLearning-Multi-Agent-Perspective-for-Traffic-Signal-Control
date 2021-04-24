from flask import Flask, request, jsonify
import numpy as np

from memory import Memory
from model import TrainModel

app = Flask(__name__)

#Random agent hyperparameters
num_layers = 4
width_layers = 480
batch_size = 100
learning_rate = 0.001
training_epochs = 800
num_states = 320
num_actions = 4

memory_size_min = 600
memory_size_max = 50000

#Left intersection agent
model_1 = TrainModel(
    num_layers, 
    width_layers, 
    batch_size, 
    learning_rate, 
    input_dim=num_states, 
    output_dim=num_actions
)
mem_1 = Memory(
   memory_size_max, 
   memory_size_min
)

#Right intersection agent
model_2 = TrainModel(
    num_layers, 
    width_layers, 
    batch_size, 
    learning_rate, 
    input_dim=num_states, 
    output_dim=num_actions
)
mem_2 = Memory(
   memory_size_max, 
   memory_size_min
)


@app.route('/initialize_agent', methods=['POST'])
def initialize_agents():
    
    #First agent
    model_1._num_layers =  request.get_json()['num_layers']
    model_1._width = request.get_json()['width_layers']
    model_1._batch_size = request.get_json()['batch_size']
    model_1._learning_rate = request.get_json()['learning_rate']
    model_1._input_dim = request.get_json()['num_states']
    model_1._output_dim = request.get_json()['num_actions']
    
    mem_1._size_max = request.get_json()['memory_size_max']
    mem_1._size_min = request.get_json()['memory_size_min']
    
    #Second agent
    model_2._num_layers =  request.get_json()['num_layers']
    model_2._width = request.get_json()['width_layers']
    model_2._batch_size = request.get_json()['batch_size']
    model_2._learning_rate = request.get_json()['learning_rate']
    model_2._input_dim = request.get_json()['num_states']
    model_2._output_dim = request.get_json()['num_actions']
    
    mem_2._size_max = request.get_json()['memory_size_max']
    mem_2._size_min = request.get_json()['memory_size_min']
    
    return "ok"

@app.route('/add_samples', methods=['POST'])
def add_sample():
    old_state_one = np.array(request.get_json()['old_state_one'])
    old_action_one = request.get_json()['old_action_one']
    reward_one = request.get_json()['reward_one']
    current_state_one = np.array(request.get_json()['current_state_one'])
    mem_1.add_sample((old_state_one, old_action_one, reward_one, current_state_one))
    
    old_state_two = np.array(request.get_json()['old_state_two'])
    old_action_two = request.get_json()['old_action_two']
    reward_two = request.get_json()['reward_two']
    current_state_two = np.array(request.get_json()['current_state_two'])
    mem_2.add_sample((old_state_two, old_action_two, reward_two, current_state_two))
    return "ok"

@app.route('/predict', methods=['POST'])
def predict():
    num = request.get_json()['num']
    if num == 1:
        model = model_1
    elif num == 2:
        model = model_2
    else:
        print("Error only 2 agents are involved (indices from 1 to 2)")
    state = np.array(request.get_json()['state'])
    prediction = model.predict_one(state)
    return jsonify(prediction=prediction.tolist())


@app.route('/replay', methods=['POST'])
def replay():
    
    num_states = request.get_json()['num_states']
    num_actions = request.get_json()['num_actions']
    gamma = request.get_json()['gamma']
    num_agent = request.get_json()['num_agent']
    
    if num_agent == 1:
        model = model_1
        mem = mem_1
    elif num_agent == 2:
        model = model_2
        mem = mem_2
    else:
        print('Error only 2 agents are involved. Index must be only 1 or 2')
    
    batch = mem.get_samples(model.batch_size)

    if len(batch) > 0:  # if the memory is full enough
        states = np.array([val[0] for val in batch])  # extract states from the batch
        next_states = np.array([val[3] for val in batch])  # extract next states from the batch

        # prediction
        q_s_a = model.predict_batch(states)  # predict Q(state), for every sample
        q_s_a_d = model.predict_batch(next_states)  # predict Q(next_state), for every sample

        # setup training arrays
        x = np.zeros((len(batch), num_states))
        y = np.zeros((len(batch), num_actions))

        for i, b in enumerate(batch):
            state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
            current_q = q_s_a[i]  # get the Q(state) predicted before
            current_q[action] = reward + gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
            x[i] = state
            y[i] = current_q  # Q(state) that includes the updated action value

        model.train_batch(x, y)  # train the NN
    return jsonify(loss=model._training_loss)

@app.route('/save_models', methods=['POST'])
def save_model():
    path = request.get_json()['path']
    model_1.save_model(path, 1)
    model_2.save_model(path, 2)
    #plot_model(model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)
    return "ok"

if __name__ == '__main__':
    # Start Web App
    app.run(threaded=False)




     