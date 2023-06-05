#!/usr/bin/env python3

"""
Created 7/11/2022

@author: Malachy Guzman

This code creates the frame-wise area/aspect to correct the issue from experiment 4
"""

import numpy as np
import pandas as pd
import sys, time

def main(): 
        #Argument is the moments csv
        filename = sys.argv[1]
        print("\n")
        print("Loaded " + filename.replace("code/",""))

        moment_df = pd.read_csv(filename, encoding = 'utf-8')

        fulltime_start = time.time()

        moment_df['normal_area*aspect'] = moment_df['seg_area']*moment_df['aspect_w/l']

        fulltime_end = time.time()
        processtime = fulltime_end-fulltime_start #in seconds
        print("Took " + str(processtime) + " sec to do calculations.") #in s
        
        #Write dataframe to csv
        savename = filename.replace("code/","")
        savename = savename.replace(".avi.csv","")
        moment_df.to_csv(savename + "_normalized_moments.csv")
        print('Done: exported normalized moment data to csv')
        print("\n")
    


if __name__ == "__main__":
    main()