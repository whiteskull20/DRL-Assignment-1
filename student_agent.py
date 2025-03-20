# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
with open("q_table.pkl", "rb") as file:
    q_table = pickle.load(file)
def get_action(obs):
    def mystate(state):
        newstate = []
        for i in range(4):
            newstate.append(int(state[0] < state[2*i+2]))
            newstate.append(int(state[0] > state[2*i+2]))
            newstate.append(int(state[1] < state[2*i+3]))
            newstate.append(int(state[1] > state[2*i+3]))
        for s in state[10:]:
            newstate.append(s)
        return tuple(newstate)
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    
    state = mystate(obs)
    if state not in q_table:
        action = np.random.choice(6)
    else:
        maxval = np.max(q_table[state])
        action = np.random.choice([a for a, val in enumerate(q_table[state]) if val == maxval])
    return action

