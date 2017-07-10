# import essential python packages
import numpy as np
import pandas as pd

import os, sys, json, requests, pickle
from urllib.request import urlretrieve

from math import sqrt, sin, cos, atan2, pi
from haversine import haversine
from datetime import datetime
from calendar import timegm

import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from scipy import stats

import warnings
warnings.filterwarnings('ignore')

# define preprocessing functions and other helper functions

# compute proximity of pickup and dropoff points with airport and label if within threshold value (< 1 mile).
def isNearAirport(row):
    #Pickup, Dropoff
    points = [(round(row['Pickup_latitude'], 4) , round(row['Pickup_longitude'], 4)), (round(row['Dropoff_latitude'], 4), round(row['Dropoff_longitude'], 4))]
    #JFK, EWR, LGA lat lon pairs
    airports = [(40.6442, -73.7822), (40.6897, -74.1745), (40.7747, -73.8719)]
    labels = []
    
    # Compute distance of each point from the list of airports and identify if its less than 1 mile.
    # Label this point with index of corresponding airport
    # 0 - Not an airport, 1 - JFK, 2 - EWR, 3 - LGA
    for pt in points:
        distances = []
        for apt in airports:
            distances.append(haversine(pt, apt, miles=True))
        min_dist = np.min(distances) 
        label    = distances.index(min_dist)
        labels.append((label+1) if (min_dist < 1) else 0)

    return labels

# convert provided value with given string representation into datetime object
def str_to_DateTime(val):
    return datetime.strptime(val, '%Y-%m-%d %H:%M:%S')

# extact pickup-hour, pickup-day of the month (1-30), trip time in hours and average speed in miles/hour for given row.
def dist_features(row):
    labels = isNearAirport(row)
    
    d1 = str_to_DateTime(row['lpep_pickup_datetime'])
    d2 = str_to_DateTime(row['Lpep_dropoff_datetime'])
    Pickup_hr = d1.hour
    Pickup_dom = d1.day
    Trip_time = (((d2 - d1).total_seconds())/3600)
    Average_speed = (0 if(Trip_time == 0) else (row['Trip_distance']/Trip_time))
    
    return pd.Series({'c1': labels[0], 
                      'c2': labels[1],
                      'c3': Pickup_hr,
                      'c4': Pickup_dom,
                      'c5': Average_speed
                     }).astype('int')

# extact pickup-hour, dropoff-hour, pickup-day, dropoff-day, trip time in hours for given row.
def process(row):
    labels = isNearAirport(row)
    d1 = str_to_DateTime(row['lpep_pickup_datetime'])
    d2 = str_to_DateTime(row['Lpep_dropoff_datetime'])
    
    Pickup_hr   = d1.hour
    Dropoff_hr  = d2.hour
    Pickup_day  = d1.weekday()
    Dropoff_day = d2.weekday()
    Trip_time   = ((d2 - d1).total_seconds())/3600

    return pd.Series({'c1': labels[0], 
                      'c2': labels[1],
                      'c3': Pickup_hr,
                      'c4': Dropoff_hr,
                      'c5': Pickup_day,
                      'c6': Dropoff_day,
                      'c7': Trip_time
                     }).astype('int')

# Derive different binned and transformed features from date
def time_features(dt):
    dt = str_to_DateTime(dt)
    
    minutes_per_bin = int((24 / float(96)) * 60)   
    num_minutes = ((dt.hour * 60) + dt.minute)
    time_bin = num_minutes / minutes_per_bin
    hour_bin = num_minutes / 60
    min_bin = (time_bin * minutes_per_bin) % 60
    time_num = (((hour_bin * 60) + min_bin + minutes_per_bin) / 2.0) / (60 * 24)
    time_cos = cos(time_num * 2 * pi)
    
    return pd.Series({'c8': int(time_bin), 
                      'c9': time_num, 
                      'c10': time_cos
                     })