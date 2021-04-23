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
from myDQN import DQN, ReplayBuffer, CNNDQN

import time
def play_game(model):
    done = False
    state = env.reset()
    while(not done):
        action = model.act(state, epsilon_final, env, device)
        next_state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.03)
        state = next_state
def play_game2(model):
    done = False
    obs = env.reset()
    # convert observation state to image state, and reshape as 1D size
    state = get_state(obs).view(image_size * image_size)
    #################################################################
    while(not done):
        action = model.act(state, epsilon_final, env, device)
        next_obs, reward, done, _ = env.step(action)
        # convert observation state to image state, and reshape as 1D size
        next_state = get_state(next_obs).view(image_size * image_size)
        #################################################################
        env.render()
        time.sleep(0.03)
        state = next_state
def play_game_CNN(model):
    done = False
    obs = env.reset()
    state = get_state2(obs)
    while(not done):
        action = model.act(state, epsilon_final,env,device)
        next_obs, reward, done, _ = env.step(action)
        next_state = get_state2(next_obs)
        env.render()
        time.sleep(0.1)
        state = next_state


import torchvision.transforms as T
from PIL import Image
image_size = 84

# transform = T.Compose([T.ToPILImage(),  # from tensors to image data
#                        T.Grayscale(num_output_channels=1), # convert to grayscale with 1 channel
#                        T.Resize((image_size, image_size), interpolation=Image.CUBIC), # resize to 84*84 by using Cubic interpolation
#                        T.ToTensor()]) # convert back to tensor

# def get_state(observation):
#     # Numpy: Use transpose(a, argsort(axes)) to invert the transposition of tensors when using the axes keyword argument.
#     # Example: x = np.ones((1, 2, 3))
#     # np.transpose(x, (1, 0, 2)).shape --> (2, 1, 3)
#     state = observation.transpose((2,0,1))
#     state = torch.from_numpy(state)
#     state = transform(state)
#     return state

transform2 = T.Compose([T.ToPILImage(),
                       T.Resize((image_size, image_size), interpolation=Image.CUBIC),
                       T.ToTensor()])

# Convert to RGB image (3 channels)

def get_state2(observation):
    state = observation.transpose((2,0,1))
    state = torch.from_numpy(state)
    state = transform2(state)
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

model = CNNDQN(3, env.action_space.n).to(device)
    
model.load_state_dict(torch.load('checkpoints/spaceInvaders-cnndqn.pth', map_location=torch.device('cpu') ),)
model.eval()

# replay_buffer = ReplayBuffer(1000)

play_game_CNN(model)
time.sleep(3)
env.close()