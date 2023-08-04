#!/usr/bin/env python3

"""
@author: Malachy Guzman

This code finds the relative standard deviation of area (A_px) and eccentric area (A_e) for each video.

IMPORTANT: Filepaths may need to be edited to work for external users.
"""

import numpy as np
import pandas as pd
import os

def filenameCleaner(filelist):
    for i in range(len(filelist)):
        filelist[i] = filelist[i].replace("-", "_").replace(".avi.csv", "").strip()
    return filelist

def main(): 
    # THIS PATH WILL NEED TO BE EDITED BY EXTERNAL USERS
    momentPath = "../moments_data/full_survey_moments/"
    dir_list = os.listdir(momentPath)

    print("looking in " + momentPath)

    df = pd.DataFrame(columns = ['filename','rsd_area',"rsd_eccenxarea"])

    for i in range(len(dir_list)):
        file = pd.read_csv(momentPath + dir_list[i])
        file = file.dropna()
        file = file.reset_index(drop = True)

        ### ONLY LOOKING AT FIRST 55 MIN OF FRAMES, assuming 30 fps
        file = file[file["frame"] <= (55*60*30)]

        # Calculate and append relative sd for each video, calculated as SD(area)/mean(area)
        newdf = pd.DataFrame()
        newdf['filename'] = [dir_list[i]]
        rsd = [np.std(file['seg_area']) / np.mean(file['seg_area'])]
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