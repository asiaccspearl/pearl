import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import warnings
from z3 import *
from QLearning import *
warnings.filterwarnings("ignore")


#####################################
########## Fixed Parameters #########
#####################################
NUM_ZONES     = 5
NUM_TIMESLOTS = 1440

class ActualModel:
    """
    A threat analytics for convex-hull representation of the clustering models (i.e., DBSCAN, K-Means)

    Attributes:
        name (str): The name of the calculator.
    """
    
    def __init__(self, dataset, num_timeslots, num_zones, eps, min_samples):
        """
        Initializes a new approximate model instance

        Parameters:
            dataframe (DataFrame): Dataset to train the cluster models
        """
        self.dataset = dataset
        self.num_timeslots = num_timeslots
        self.num_zones = num_zones
        self.eps = eps
        self.min_samples = min_samples
        
    def clustering_dbscan(self, data):
        """
        DBSCAN clustering from given data samples and specified model parameters

        Parameters:
            data (np array): Data features (i.e., from complete or partial dataset) to train the cluster models
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
            min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point
        
        Returns:
            sklearn model: Clustering (i.e., sklearn.cluster.DBSCAN) model
        """
        db = DBSCAN(eps = self.eps, min_samples = self.min_samples)
        cluster = db.fit(data)

        return cluster


    def get_clusters(self):
        clusters = []

        for zone in range(self.num_zones):
            
            mod_dataframe = pd.DataFrame()
        
            for i in range(len(self.dataset)):
                if int(self.dataset['Occupant\'s Zone'][i] == zone):
                    mod_dataframe = mod_dataframe.append(self.dataset.loc[i, ['Zone Arrival Time (Minute)', 'Stay Duration (Minute)']])
        
            try:    
                features = np.empty([len(mod_dataframe), 2])
                features[:, 0:1] = mod_dataframe.loc[:, ['Zone Arrival Time (Minute)']].values
                features[:, 1:] = mod_dataframe.loc[:, ['Stay Duration (Minute)']].values
            except:
                continue
            
            cluster_model = DBSCAN(eps = self.eps, min_samples = self.min_samples).fit(features)
            labels = cluster_model.labels_
            core_sample_indexes = cluster_model.core_sample_indices_
        
            # Get the number of clusters
            num_clusters = len(set(cluster_model.labels_)) - (1 if -1 in cluster_model.labels_ else 0)
        
            zone_clusters = []
            #for i in range(len(core_sample_indexes)):
            #    print(labels[core_sample_indexes[i]])
        
            for i in range(num_clusters):
                arr = []
                for j in range(len(core_sample_indexes)):
                    if labels[core_sample_indexes[j]] == i:
                        index = core_sample_indexes[j]
                        arrival = features[index][0]
                        stay = features[index][1]
                        
                        #if plot_circle_line_intersection(eps, arrival, stay, 546):
                            
                            
                        #print(eps, arrival, stay, plot_circle_line_intersection(eps, arrival, stay, 546))
                        arr.append((arrival, stay))
                       
                zone_clusters.append(arr)
                
            clusters.append(zone_clusters)
            #draw_circles(zone, zone_clusters)
        return clusters

                    
    def circle_line_intersection(self, radius, center_x, center_y, line_x):
        # Generate y-coordinates of the intersection points
        if np.abs(center_x - line_x) <= radius:
            delta_x = np.abs(center_x - line_x)
            y1 = center_y + np.sqrt(radius**2 - delta_x**2)
            y2 = center_y - np.sqrt(radius**2 - delta_x**2)
            #print(y1,y2)
            if y1 > y2:
                temp = y1
                y1 = y2
                y2 = temp
            return y1, y2
        else:
            #print("No intersection points found!")
            return -1 
         
    def range_calculation(self):
        """
        '''''''Calculating the ranges (i.e., valid ranges) of of the cluster

        ''''''''Parameters:
            list[dict]: List dictionary (i.e., constaints zone id (int), cluster id (int), vertices)
            
        Returns:
            list[list[list]] and list[list[list]]: returns minimum and maximum valied ranges for a particular zone, time, and cluster
        """
        list_time_min = [[[] for j in range(self.num_timeslots)] for i in range(self.num_zones)]
        list_time_max = [[[] for j in range(self.num_timeslots)] for i in range(self.num_zones)]
        
        clusters = self.get_clusters()
        
        for i in range(self.num_zones):
            for j in range(self.num_timeslots):
                try:
                    current_cluster = clusters[i]
                except:
                    continue
                for cluster in current_cluster:
                    max_point = -1
                    min_point = 1500
                    for circle in cluster:
                        if self.circle_line_intersection(self.eps, circle[0], circle[1], j) != -1:
                            #print("c", i, j, circle[0], circle[1], plot_circle_line_intersection(eps, circle[0], circle[1], j))
                            if self.circle_line_intersection(self.eps, circle[0], circle[1], j)[0] >= 0:
                                min_point = min(min_point, self.circle_line_intersection(self.eps, circle[0], circle[1], j)[0])
                            else:
                                min_point = 0
        
                            if self.circle_line_intersection(self.eps, circle[0], circle[1], j)[1] >= 0:
                                max_point = max(max_point, self.circle_line_intersection(self.eps, circle[0], circle[1], j)[1])
                            else:
                                max_point = 0
                                                
                    if min_point != 1500:
                        if j + min_point > 1440:
                            list_time_min[i][j].append(1440 - j)
                        else:
                            list_time_min[i][j].append(int(min_point))
                    if max_point != -1:
                        if j + max_point > 1440:
                            list_time_max[i][j].append(1440 - j)
                        else:
                            list_time_max[i][j].append(int(max_point)) 
        return list_time_min, list_time_max

    
    def noise_augmented_range_calculation(self):
        
        list_time_min, list_time_max = self.range_calculation()
        
        num_benign = 0
        num_anomaly = 0
        data = self.dataset[['Occupant\'s Zone','Zone Arrival Time (Minute)', 'Stay Duration (Minute)']].values


        for i in range(len(data)):
            zone = int(data[i][0])
            entrance = int(data[i][1])
            duration = int(data[i][2])
            flag = False
            for j in range(len(list_time_min[zone][entrance])):
                if duration >= list_time_min[zone][entrance][j] and duration <= list_time_max[zone][entrance][j]:
                    flag = True
                    num_benign +=1
            if flag == False:
                num_anomaly += 1
                list_time_min[zone][entrance].append(duration)
                list_time_max[zone][entrance].append(duration)
        return list_time_min, list_time_max
                

    def threat_analytics(self, knowledge, accessibleZones):
        """
        Adds two numbers and returns the result.

        Parameters:
            knowledge (str): The percentage of data used to train models accessible to the attackers
            accessibleZones (list[int]): The set if zones exploitable by the attackers

        Returns:
            list[Int]: Attack schedule
        """
        attack_schedule = np.array()
        list_time_min, list_time_max = self.noise_augmented_range_calculation()
        
        return attack_schedule
    

# =============================================================================
# dataset_house_A_occ_1 = pd.read_csv("../../data/cleaned/Cleaned-Dataframe_House-A_Occupant-1.csv")  
# dataset_house_A_occ_2 = pd.read_csv("../../data/cleaned/Cleaned-Dataframe_House-A_Occupant-2.csv")  
# dataset_house_B_occ_1 = pd.read_csv("../../data/cleaned/Cleaned-Dataframe_House-B_Occupant-1.csv")  
# dataset_house_B_occ_2 = pd.read_csv("../../data/cleaned/Cleaned-Dataframe_House-B_Occupant-2.csv")  
# 
# =============================================================================









