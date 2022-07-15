#!/usr/bin/env python3

"""
Created 7/11/2022

@author: Malachy Guzman

This code compares the output of moment_analysis.py to find average and median linear fits and R^2's 
for each metric across sample videos.
"""

from doctest import testmod
import ReadNPYAppend as r
import numpy as np
import pandas as pd
import os
import sys, time, csv, argparse
#from scipy import stats

       
path = "../moments_data/full_kbatch_mom/stats/"
os.chdir(path)

stat_df = np.zeros((6,4))

# Read CSV's and create 3d stat_df
for file in os.listdir():
    if file.endswith(".csv"):
        with open(file, 'r') as f:
            sample = np.array(pd.read_csv(file))
            stat_df = np.dstack([stat_df, sample])

stat_df = np.delete(stat_df, 0, axis=2)
mean_df = np.copy(stat_df[:,:,1])


for i in range(1, stat_df.shape[0]):
    for j in range(1, stat_df.shape[1]):
        mean_df[i][j] = np.mean(np.abs(stat_df[i,j,:]))


r_sq_stdev = np.std(stat_df[:,3,:], axis=1, dtype = np.float64)

mean_df = pd.DataFrame(mean_df)
mean_df.columns = ['metric', 'slope', 'int', "R^2"]
mean_df["R^2 SD"] = r_sq_stdev

# Mean
print("\nHighest mean slope:")
print(mean_df.sort_values(by=['slope'], ascending=False))
print("\nHighest mean R^2:")
print(mean_df.sort_values(by=['R^2'], ascending=False))
print("\n")



#Median
# median_df = pd.DataFrame(median_df)
# median_df.columns = ['metric', 'slope', 'int', "R^2"]
# print("\nHighest median slope:")
# print(median_df.sort_values(by=['slope'], ascending=False))
# print("\nHighest median R^2:")
# print(median_df.sort_values(by=['R^2'], ascending=False))
