import glob
import torch.cuda as tc
import torch 

from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import gym
from gym import spaces
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import fnss 
import networkx as nx
from os import path 

path1 = Path('analysis.py')
path2 = str(path1.parent.absolute().parent.absolute()) + '/logs'
print(path2)
files = glob.glob(path2+'/'+'*STATE.txt')

# cols = ["node_memory0","dist_src0","dist_serving0","dist_rec0","lifetime_avg0","node_memory1","dist_src1","dist_serving1","dist_rec1","lifetime_avg1","node_memory2","dist_src2","dist_serving2","dist_rec2","lifetime_avg2","node_memory3","dist_src3","dist_serving3","dist_rec3","lifetime_avg3","node_memory4","dist_src4","dist_serving4","dist_rec4","lifetime_avg4","node_memory5","dist_src5","dist_serving5","dist_rec5","lifetime_avg5","node_memory6","dist_src6","dist_serving6","dist_rec6","lifetime_avg6","node_memory7","dist_src7","dist_serving7","dist_rec7","lifetime_avg7","node_memory8","dist_src8","dist_serving8","dist_rec8","lifetime_avg8","node_memory9","dist_src9","dist_serving9","dist_rec9","lifetime_avg9","node_memory10","dist_src10","dist_serving10","dist_rec10","lifetime_avg10","node_memory11","dist_src11","dist_serving11","dist_rec11","lifetime_avg11","node_memory12","dist_src12","dist_serving12","dist_rec12","lifetime_avg12","node_memory13","dist_src13","dist_serving13","dist_rec13","lifetime_avg13","node_memory14","dist_src14","dist_serving14","dist_rec14","lifetime_avg14","node_memory15","dist_src15","dist_serving15","dist_rec15","lifetime_avg15","node_memory16","dist_src16","dist_serving16","dist_rec16","lifetime_avg16","node_memory17","dist_src17","dist_serving17","dist_rec17","lifetime_avg17","node_memory18","dist_src18","dist_serving18","dist_rec18","lifetime_avg18","node_memory19","dist_src19","dist_serving19","dist_rec19","lifetime_avg19","node_memory20","dist_src20","dist_serving20","dist_rec20","lifetime_avg20","node_memory21","dist_src21","dist_serving21","dist_rec21","lifetime_avg21","node_memory22","dist_src22","dist_serving22","dist_rec22","lifetime_avg22","node_memory23","dist_src23","dist_serving23","dist_rec23","lifetime_avg23","node_memory24","dist_src24","dist_serving24","dist_rec24","lifetime_avg24","node_memory25","dist_src25","dist_serving25","dist_rec25","lifetime_avg25","node_memory26","dist_src26","dist_serving26","dist_rec26","lifetime_avg26","node_memory27","dist_src27","dist_serving27","dist_rec27","lifetime_avg27","node_memory28","dist_src28","dist_serving28","dist_rec28","lifetime_avg28","node_memory29","dist_src29","dist_serving29","dist_rec29","lifetime_avg29","node_memory30","dist_src30","dist_serving30","dist_rec30","lifetime_avg30","node_memory31","dist_src31","dist_serving31","dist_rec31","lifetime_avg31","node_memory32","dist_src32","dist_serving32","dist_rec32","lifetime_avg32","node_memory33","dist_src33","dist_serving33","dist_rec33","lifetime_avg33","node_memory34","dist_src34","dist_serving34","dist_rec34","lifetime_avg34","node_memory35","dist_src35","dist_serving35","dist_rec35","lifetime_avg35","node_memory36","dist_src36","dist_serving36","dist_rec36","lifetime_avg36","node_memory37","dist_src37","dist_serving37","dist_rec37","lifetime_avg37","node_memory38","dist_src38","dist_serving38","dist_rec38","lifetime_avg38","node_memory39","dist_src39","dist_serving39","dist_rec39","lifetime_avg39","source_pop", "packet_size", "packet_life", "hit_type", "serving_node", "penalty","y"]
# print(len(cols))
# final_df = pd.read_csv(files[0], delimiter='\t', header=0, usecols=my_cols)
# print(final_df)
final_df = pd.DataFrame()
files_to_be_included = 5
# for file in files[:files_to_be_included]:

#     data = pd.read_csv(file, delimiter='\t')
#     data=data.iloc[1:]
#     final_df = pd.concat([final_df, data.reset_index()], axis=0, ignore_index=True)
#     final_df.replace([-np.inf,np.inf,np.nan],0, inplace=True)
    
# for col in final_df.columns:
#     final_df[col].astype('int')

# Example Dataset
class CustomDataset(Dataset):
    def __init__(self, files):
        final_df = pd.DataFrame()
        for file in files:
            data = pd.read_csv(file, delimiter='\t')
            final_df = pd.concat([final_df, data.reset_index()], axis=0, ignore_index=True)

        final_df.fillna(0,inplace=True)
        self.data = final_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        next_row = self.data.iloc[idx].to_list()[:-1]
        return next_row

# Custom Environment
class CustomEnv(gym.Env):
    def __init__(self, data_loader, state_size, action_size):
        super(CustomEnv, self).__init__()
        
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(action_size)
        
        self.state = None
        self.current_step = 0
    
    def reset(self):
        self.data_iter = iter(self.data_loader)
        self.state = self._get_next_state()
        self.current_step = 0
        return self.state
    
    def step(self, action):
        # reward = self._calculate_penalty(action)
        
        self.state = self._get_next_state()
        self.current_step += 1
        reward = self.state[-1]
        done = self.state is None
        info = {}
        
        return self.state, reward, done, info
    
    def _get_next_state(self):
        try:
            state = next(self.data_iter)
            return state
        except StopIteration:
            return None
    
    def _calculate_penalty(self, action):
        penalty = np.sum((self.state - action) ** 2)
        return -penalty
    
    def render(self, mode='human'):
        print(f'Step: {self.current_step} State: {self.state}')
    
    def close(self):
        pass

# # Example data
# # data = np.random.rand(100, 10)  # 100 samples with 10 features each



# Create Dataset and DataLoader
dataset = CustomDataset(files[:files_to_be_included])
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Create the environment
state_size = 207
action_size = 3
env = CustomEnv(data_loader, state_size, action_size)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

policy_net = DQN(state_size=state_size, action_size=action_size)
target_net = DQN(state_size=state_size, action_size=action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()


gamma = 0.99
batch_size = 64
epsilon = 0.1
memory_capacity = 10000
memory = ReplayMemory(memory_capacity)
scenarios_dir = str(path1.parent.absolute().parent.absolute())
topology = fnss.parse_topology_zoo(path.join(scenarios_dir, 'scenarios/resources/test.graphml')).to_undirected()

def select_action(state, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).argmax().view(1, 1)
    else:
        return torch.tensor([[random.randrange(3)]], dtype=torch.long)
    
def compute_path_length(source,action):

    edge = []
    root = []
    intermediate = []
    for node in topology.nodes():
        if topology.nodes[node]['label'] == 'edge router':
            edge.append(node)
        elif topology.nodes[node]['label'] == 'root router':
            root.append(node)
        elif topology.nodes[node]['label'] == 'int router':
            intermediate.append(node)

    caches = [edge, intermediate, root]
    layer = caches[action]
    cache_node = random.choice(layer)

    return nx.shortest_path_length(topology, source,cache_node)

def compute_reward(hit_type, packet_size, source, action):

    transmission_energy = 15
    sense_energy = 50
    activate_energy = 150

    path_length = compute_path_length(source,action)
    
    return packet_size *transmission_energy*path_length
    
def get_states(batch):
    pass 


def optimize_model():
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))

    state_b = batch[0][35]

    state_batch = state_b
    action_batch = batch[1][0]
    reward_batch = batch[2][0]
    next_state_batch = state_b
    done_batch = batch[4][0]
    state_action_values = max(policy_net(state_batch))
    
    next_state_values = max(target_net(next_state_batch))
    expected_state_action_values = (next_state_values * gamma * (1 - done_batch)) + reward_batch
    
    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

num_episodes = 100
for i_episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    
    for t in range(1000):

        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.tensor([next_state], dtype=torch.float32) if next_state is not None else torch.zeros_like(state)
        ns = next_state[0]
        reward = compute_reward(ns[-4],ns[-6],int(ns[-3]),action)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)
        memory.push((state, action, reward, next_state, done))
        state = next_state
        
        optimize_model()
        
        if done:
            break
    
    if i_episode % 10 == 0:
        print('target copied')
        target_net.load_state_dict(policy_net.state_dict())

env.close()

