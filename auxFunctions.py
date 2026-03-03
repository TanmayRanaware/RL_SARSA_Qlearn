import numpy as np
import gymnasium as gym
import pickle

# Create environment (only used for action space size)
env = gym.make('MountainCar-v0')

# Discretize observation space
pos_space = np.linspace(-1.2, 0.6, 18)
vel_space = np.linspace(-0.07, 0.07, 28)

def getState(observation):
    pos, vel = observation
    pos_bin = np.digitize(pos, pos_space)
    vel_bin = np.digitize(vel, vel_space)
    return (pos_bin, vel_bin)

def createEmptyQTable():
    Q = {}

    for pos in range(len(pos_space) + 1):
        for vel in range(len(vel_space) + 1):
            state = (pos, vel)
            for action in range(env.action_space.n):
                Q[(state, action)] = 0

    return Q

def maxAction(Q, state, actions=[0, 1, 2]):
    values = np.array([Q[(state, a)] for a in actions])
    action = np.argmax(values)
    return action

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)