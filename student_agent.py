# Remember to adjust your student ID in meta.xml
import numpy
import pickle
import random
import gym

with open("q_table4.pkl", "rb") as file:
    q_table = pickle.load(file)
prev_action = -1
prev_passenger = 0
prev_destination = 0
visited = {}
def mystate(state,prev_action,prev_passenger,prev_destination,visited):
      newstate = []
      for s in state[10:]:
          newstate.append(s)
      newstate.append(prev_action)
      newstate.append(prev_passenger)
      newstate.append(prev_destination)
      newstate.append(int((state[0],state[1]) in visited))
      if prev_action != -1:
        visited[(state[0],state[1])] = 1
      return tuple(newstate),visited
def get_action(obs):
        # TODO: Train your own agent
        # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
        # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
        #       To prevent crashes, implement a fallback strategy for missing keys. 
        #       Otherwise, even if your agent performs well in training, it may fail during testing.
        global prev_passenger
        global prev_destination
        global prev_action
        global visited
        state,visited = mystate(obs,prev_action,prev_passenger,prev_destination,visited)
        if state not in q_table:
            q_table[state] = [0]*6
        q_value = [q_table[state][a] for a in range(6)]
        maxval = numpy.max(q_value)
        action = numpy.random.choice([a for a, val in enumerate(q_value) if val == maxval])
        prev_action = action
        prev_passenger,prev_destination = obs[14],obs[15]
        return action



