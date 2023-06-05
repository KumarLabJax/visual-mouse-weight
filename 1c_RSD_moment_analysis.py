#!/usr/bin/env python3

"""
Created 5/4/22

@author: Malachy Guzman

This code finds the relative standard deviation of area for each video
"""

import ReadNPYAppend as r
import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy import stats
import os, sys, time, csv, argparse
 

def filenameCleaner(filelist):
    for i in range(len(filelist)):
        filelist[i] = filelist[i].replace("-", "_").replace(".avi.csv", "").strip()
    return filelist


def main(): 
    momentPath = "../moments_data/full_survey_moments/"
    dir_list = os.listdir(momentPath)

    print("looking in " + momentPath)

    df = pd.DataFrame(columns = ['filename','rsd_area',"rsd_eccenxarea"])

    for i in range(len(dir_list)):
        file = pd.read_csv(momentPath + dir_list[i])
        file = file.dropna()
        file = file.reset_index(drop = True)

        ### ONLY LOOKING AT FIRST 55 MIN OF FRAMES, assuming 30 fps = 99,000 frames
        file = file[file["frame"] <= (55*60*30)]

        # Calculate and append relative sd for each video, calculated as SD(area)/mean(area)
        newdf = pd.DataFrame()
        newdf['filename'] = [dir_list[i]]
        rsd = [np.std(file['seg_area']) / np.mean(file['seg_area'])]
        rsd_EccArea = [np.std(file['seg_area']*file['eccentricity']) / np.mean(file['seg_area']*file['eccentricity'])]
        rsd_EccArea = [np.std(file['seg_area']*file['eccentricity']) / np.mean(file['seg_area']*file['eccentricity'])]
        newdf['rsd'] = rsd
        newdf['rsd_EccArea'] = rsd_EccArea

        df = pd.concat([df, newdf], ignore_index = True)

        print("finished video " + str(i) + " with ecc rsd " + str(rsd_EccArea))


    print("\nDone calculating moment medians")

    savename = "rsd_ecc_fullstrainsurvey"
    df.to_csv(savename + ".csv")
    print('Done exporting RSDs to CSV')
    
if __name__ == "__main__":
    main()