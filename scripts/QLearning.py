import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import warnings
from z3 import *
warnings.filterwarnings("ignore")

import copy

class QLearning:

    def __init__(self, num_timeslots, num_zones, list_time_min, list_time_max, num_episodes, num_iterations, epsilon, learning_rate, discount_factor):        
        
        self.num_timeslots = num_timeslots
        self.num_zones = num_zones
        self.list_time_min = list_time_min
        self.list_time_max = list_time_max
        self.num_episodes = num_episodes
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = dict()
        self.next_states = dict()
        self.prev_states = dict()
        self.final_states = []
        
    def get_rewards(self):
        
        return [0, 1, 2, 4, 3]
    
    def get_sorted_duration(self, arrival_time, arrival_zone):
        
        sorted_durations = set()
        
        for cluster in range(len(self.list_time_min[arrival_zone][arrival_time])):
            for duration in range(self.list_time_min[arrival_zone][arrival_time][cluster], self.list_time_max[arrival_zone][arrival_time][cluster] + 1):
                sorted_durations.add(int(duration))
        sorted_durations = list(sorted_durations)
        sorted_durations = sorted(sorted_durations)
        if 0 in sorted_durations:
            sorted_durations.remove(0)
        return sorted_durations
    
    
    def get_all_possible_states(self):
        
        states = ['*-*-*']
        
        for arrival_time in range(self.num_timeslots):
            for arrival_zone in range(self.num_zones):
                stay_durations = self.get_sorted_duration(arrival_time, arrival_zone)
                for stay_duration in stay_durations:
                    if stay_duration > 0:
                        states.append(str(arrival_time) + '-' + str(arrival_zone) + '-' + str(stay_duration))
        return states
    
    
    def get_all_possible_actions(self):
        
        actions = []
        
        for zone in range(self.num_zones):
            actions.append(zone)
        return actions
    
    
    def init_q_table(self):
        
        states = self.get_all_possible_states()
        actions = self.get_all_possible_actions()
        
        for state in range(len(states)):
            for action in range(len(actions)):
                self.q_table[states[state] + '_' + str(actions[action])] = 0
    
    
    def get_next_states(self, state):
        
        if state == '*-*-*':
            next_states = []
            
            for i in range(self.num_zones):
                arrival_zone_time_stay_durations = self.get_sorted_duration(0, i)
                if len(arrival_zone_time_stay_durations) == 0:
                    next_states.append(-1)
                else:
                    next_states.append(str(0) + '-' + str(i) + '-' + str(arrival_zone_time_stay_durations[0]))
            return next_states
        
        
        
        arrival_time = int(state.split('-')[0])
        arrival_zone = int(state.split('-')[1])
        state_stay_duration = int(state.split('-')[2])
    
        next_states = []
        for i in range(self.num_zones):
            next_states.append(-1)
    
        ##############################################################
        ################ determine current state #####################
        ##############################################################
        
        arrival_zone_time_stay_durations = self.get_sorted_duration(arrival_time, arrival_zone)
        
        for stay_duration in range (len(arrival_zone_time_stay_durations) - 1):
            if arrival_zone_time_stay_durations[stay_duration] == state_stay_duration:
                next_state = str(arrival_time) + '-' + str(arrival_zone) + '-' + str(arrival_zone_time_stay_durations[stay_duration + 1])
                next_states[arrival_zone] = next_state
        
        actions = self.get_all_possible_actions()
        
        for action in actions:
            
            if action != arrival_zone:
                
                next_arrival_time = arrival_time + state_stay_duration
                
                stay_durations = self.get_sorted_duration(next_arrival_time, action)
                
                if len(stay_durations) != 0:
                    
                    next_arrival_zone = action
                    next_stay_duration = stay_durations[0]
                    if next_stay_duration != 0:
                        next_state = str(next_arrival_time) + '-' + str(next_arrival_zone) + '-' + str(next_stay_duration)
                        next_states[action] = next_state
                
        return next_states
    
    def generate_next_prev_states(self):
        
        states = self.get_all_possible_states()
        
        for state in states:
            try:
                next_states = self.get_next_states(state)
                self.next_states[state] = next_states
                for next_state in next_states:
                    self.prev_states[next_state] = state
            except:
                pass
            
            
            
    def pop_states(self, state):
        
        while True:
            next_states_state = self.next_states[state]
            
            valid_states = []
            
            for i in range(len(next_states_state)):
                if next_states_state[i] != -1:
                    valid_states.append(i)
          
            if len(valid_states) == 0:
                all_actions = self.get_all_possible_actions()
                
                for action in range(len(all_actions)):
                    self.q_table.pop(state + '_' + str(action))

                self.next_states.pop(state)
                
                state_zone = int(state.split('-')[1])
                current_state = copy.copy(state)
                
                try:
                    state = self.prev_states[state]
                    self.next_states[state][state_zone] = -1                       
                    self.prev_states.pop(current_state)

                except:
                    break
            
            else:
                return
    
    def model_training(self):
        
        self.generate_next_prev_states()
        
        self.init_q_table()
        
        total_costs      = []
        attack_schedules = []
        for episode in range(self.num_episodes):
        
            for iteration in range(self.num_iterations):
                
                total_cost = 0
                timeslot = 0
                state = '*-*-*'
                
                while True:  

                    valid_actions = []
        
                    for i in range(len(self.next_states[state])):
                        if self.next_states[state][i] != -1:
                            valid_actions.append(i)
                    
                    ############################## Exploration #####################################
                    if np.random.random() >= self.epsilon and len(valid_actions) > 1:
                        action = valid_actions[np.random.randint(len(valid_actions))]
         
                    ############################## Exploitation ####################################    
                    else:
                        if len(valid_actions) > 0:
                            action = valid_actions[0]
                    
                            action = valid_actions[0]
                            max_q_val = self.q_table[state + '_' + str(valid_actions[0])]
            
                            for i in range(1, len(valid_actions)):
                                q_val = self.q_table[state + '_' + str(valid_actions[i])]
            
                                if q_val > max_q_val:
                                    max_q_val = q_val
                                    action = valid_actions[i]
                        else:
                            self.pop_states(state)
                            break
        
        
                    current_q_val = self.q_table[state + '_' + str(action)]
                    
                    #print(state)
                    
                    next_state = self.next_states[state][action]
                    
                    if next_state not in self.next_states:
                        #print("state", state, "next state", next_state)
                        break
                    
                    next_next_states = self.next_states[next_state]
                        
                    next_valid_actions = []
                    
                    for i in range(len(next_next_states)):
                        if next_next_states[i] != -1:
                            next_valid_actions.append(i)
                     
                    if next_valid_actions == []:
                        self.pop_states(next_state)
                        
                        break
        
                    else:
                        next_max_q_val = self.q_table[next_state + '_' + str(next_valid_actions[0])]
                        next_action = next_valid_actions[0]
                    
                        for i in range(1, len(next_valid_actions)):
                            next_q_val = self.q_table[next_state + '_' + str(next_valid_actions[i])]
        
                            if next_q_val > next_max_q_val:
                                next_max_q_val = next_q_val
                                next_action = next_valid_actions[i]
                        
                        next_q_val = self.q_table[next_state + '_' + str(next_action)]
                        
                        
        
                        next_state_time = int(next_state.split('-')[0])
                        next_state_zone = int(next_state.split('-')[1])
                        next_state_duration = int(next_state.split('-')[2])
                        
                        
                        reward = self.get_rewards()[action]
                        
                        
                        self.q_table[state + '_' + str(action)] =  current_q_val + self.learning_rate * ( reward + self.discount_factor * next_q_val - current_q_val)
                    
                    next_arrival_time = int(next_state.split('-')[0])
                    next_stay_duration = int(next_state.split('-')[2])
                    
                    state = next_state
                    if next_arrival_time + next_stay_duration >= 1437:
                        break
            print("episode", episode, "cost", self.model_inference()[0], "q-table-size", len(self.q_table))
            total_cost, attack_schedule = self.model_inference()
        
            total_costs.append(total_cost)
            attack_schedules.append(attack_schedule)
        return total_costs, attack_schedules
            
    def model_inference(self):
        
        state = '*-*-*'
        total_cost = 0
        all_states = []
        while True:
            
            all_states.append(state)
            
            if state not in self.next_states:
                break
                
                
            next_states = self.next_states[state]
            
            valid_actions = []
        
            for i in range(len(next_states)):
                if next_states[i] != -1:
                    valid_actions.append(i)
                        
            if len(valid_actions) == 0:
                break
        
            action = valid_actions[0]
            max_q_val = self.q_table[state + '_' + str(valid_actions[0])]
            
            for i in range(1, len(valid_actions)):
                q_val = self.q_table[state + '_' + str(valid_actions[i])]
                if q_val > max_q_val:
                    max_q_val = q_val
                    action = valid_actions[i]
        
            next_state = next_states[action]
        
            next_arrival_time = int(next_state.split('-')[0])
            next_stay_duration = int(next_state.split('-')[2])
        
                
            next_state_time = int(next_state.split('-')[0])
            next_state_zone = int(next_state.split('-')[1])
            next_state_duration = int(next_state.split('-')[2])
            
            if state != '*-*-*':            
                 state_zone = int(state.split('-')[1])       
                 state_time = int(state.split('-')[0])
                 total_cost += (next_state_time - state_time) * self.get_rewards()[state_zone]
            
            state = next_state
            
            
            if next_arrival_time + next_stay_duration >= 1437:
            
                break
        
        schedule = []

        for i in range(1, len(all_states) - 1):
            zone = int(all_states[i].split('-')[1])
            next_zone = int(all_states[i + 1].split('-')[1])
            duration = int(all_states[i].split('-')[2])
            
            if zone !=  next_zone:
                for j in range(duration):
                    schedule.append(zone)
        
        for i in range(len(schedule), 1440):
            schedule.append(next_zone)
            
        
        return total_cost, schedule
            