#!/usr/bin/env python3

"""
@author: Malachy Guzman

This code compresses each video's per-frame .csv data into medians, appending values to the strain survey data and cleaning it up. 

IMPORTANT: Filepaths may need to be edited to work for external users.
"""

import ReadNPYAppend as r
import numpy as np
import pandas as pd
from scipy import stats
import os, sys, time, csv, argparse


def filenameCleaner(filelist):
    for i in range(len(filelist)):
        filelist[i] = filelist[i].replace("-", "_").replace(".avi.csv", "").strip()
    return filelist


def main(): 
    # THIS PATH WILL NEED TO BE EDITED BY EXTERNAL USERS
    momentPath = "../moments_data/full_survey_moments/"
    dir_list = os.listdir(momentPath)

    median_df = pd.DataFrame(columns = ['filename','area','aspect','eccentricity','elongation'])

    for i in range(len(dir_list)):
        file = pd.read_csv(momentPath + dir_list[i])
        #Cleaning out NA values
        file = file.dropna()
        file = file.reset_index(drop = True)

        ### ONLY LOOKING AT FIRST 55 MIN OF FRAMES, assuming 30 fps 
        file = file[file["frame"] <= (55*60*30)]

        #Adding metric medians
        newRow = pd.DataFrame() 
        newRow['filename'] = [dir_list[i]]
        newRow['area'] = [np.median(file['seg_area'])]
        newRow['aspect'] = [np.median(file['aspect_w/l'])]
        newRow['eccentricity'] = [np.median(file['eccentricity'])]
        newRow['elongation'] = [np.median(file['elongation'])]
        newRow['speed'] = [np.median(file['v_mag'])] 

        median_df = pd.concat([median_df, newRow], ignore_index = True)
        print("done with vid " + str(i))


    print("\nDone calculating moment medians")

    # Export median data in case of error later on
    savename = "median_moments_fullstrainsurvey"
    median_df.to_csv(savename + ".csv")
    print('Done exporting moment medians to csv')


    ###### Data cleaning/arranging, requires median df to already exist
    # Reading moments csv back in
    momentPath = "median_moments_fullstrainsurvey.csv"
    median_df = pd.read_csv(momentPath)
    median_df.rename(columns = {'filename':'basename'}, inplace = True)
    print("\nInit median shape: " + str(median_df.shape))

    # Read in strain survey 
    strainPath = "../StrainSurveyMetaList_2019-04-09.tsv"
    strain_survey = pd.read_csv(strainPath, sep = '\t')
    print("Init strain shape: " + str(strain_survey.shape))

    # Cleaning out rows with NA weights
    strain_survey = strain_survey[strain_survey['Weight'].notna()]
    strain_survey = strain_survey.reset_index(drop = True)

    # Fixing names
    strain_survey['basename']  = [x.split('/')[-1] for x in strain_survey['NetworkFilename'].values]
    strain_survey['basename']  = [x.replace(".avi","") for x in strain_survey['basename'].values]
    median_df['basename']  = [x.replace(".avi_moments.csv","") for x in median_df['basename'].values]
    median_df = median_df.drop(['Unnamed: 0'], axis = 1) 

    print("\n")

    # Putting strain and median together
    combinedData = pd.merge(strain_survey, median_df, how = 'outer', on = 'basename')

    # Cleaning out NA values again (Missing moments data)
    combinedData = combinedData[combinedData['area'].notna()]
    combinedData = combinedData[combinedData['aspect'].notna()]
    combinedData = combinedData[combinedData['eccentricity'].notna()]
    combinedData = combinedData[combinedData['elongation'].notna()]
    combinedData = combinedData[combinedData['speed'].notna()] # New line for cleaning speed
    combinedData = combinedData.reset_index(drop = True)

    print("\nFinal median shape: " + str(median_df.shape))
    print("Final strain shape: " + str(strain_survey.shape))
    print("Combined shape: " + str(combinedData.shape))

    combinedData['areaxeccen'] = combinedData['area']*combinedData['eccentricity']

    combinedData.to_csv("fullsurvey_momentmedians.csv")
    print("Appended moment median data to strain survey and saved as csv")


    
if __name__ == "__main__":
    main()