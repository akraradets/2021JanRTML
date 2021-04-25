import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
import gym
import numpy as np
from tqdm import trange
from myDQN import DQN, ReplayBuffer, CNNDQN,DDQN

import time

def play_game_CNN(model):
    done = False
    obs = env.reset()
    state = get_state3(obs)
    while(not done):
        action = model.act(state, epsilon_final,env,device)
        next_obs, reward, done, _ = env.step(action)
        next_state = get_state3(next_obs)
        env.render()
        time.sleep(0.1)
        state = next_state


import torchvision.transforms as T
from PIL import Image

image_size = 84
transform = T.Compose([T.ToPILImage(),
                       T.Grayscale(num_output_channels=1),
                       T.Resize((image_size, image_size), interpolation=Image.CUBIC),
                       T.ToTensor()])

# Convert to RGB image (3 channels)
import queue
state_buffer = queue.Queue()
def get_state3(observation):
    
    # First time, repeat the state for 3 times
    if(state_buffer.qsize() == 0):
        for i in range(3):
            state = get_state2(observation)
            state_buffer.put(state)
        # print(observation.shape, state.shape)
    else:
        state_buffer.get()
        state = get_state2(observation)
        state_buffer.put(state)
    # for i in state_buffer.queue:
    #     print(i.shape)
    rep = torch.cat(list(state_buffer.queue), dim=0)
    # print("rep=====",rep.shape)
    return rep

def get_state2(observation):
    state = observation.transpose((2,0,1))
    state = torch.from_numpy(state)
    state = transform(state)
    return state




# Select GPU or CPU as device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)



epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

# Epsilon annealing schedule generator

def gen_eps_by_episode(epsilon_start, epsilon_final, epsilon_decay):
    eps_by_episode = lambda episode: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * episode / epsilon_decay)
    return eps_by_episode

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
eps_by_episode = gen_eps_by_episode(epsilon_start, epsilon_final, epsilon_decay)


env_id = 'SpaceInvaders-v0'
env = gym.make(env_id)

model = DDQN(3, env.action_space.n).to(device)
    
model.load_state_dict(torch.load('checkpoints/spaceInvaders-hw-phi-50M.pth', map_location=torch.device('cpu') ),)
model.eval()

# replay_buffer = ReplayBuffer(1000)

play_game_CNN(model)
time.sleep(3)
env.close()
