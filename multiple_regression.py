#!/usr/bin/env python3

"""
Created 7/11/2022

@author: Malachy Guzman

This code compresses single vid moments into medians and appends to strain survey

This code also compares the output of moment_analysis.py to find average and median linear fits and R^2's 
for each metric across sample videos.
"""

import ReadNPYAppend as r
import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy import stats
import os, sys, time, csv, argparse
#import matplotlib.pyplot as plt
 


def filenameCleaner(filelist):
    for i in range(len(filelist)):
        filelist[i] = filelist[i].replace("-", "_").replace(".avi.csv", "").strip()
    return filelist


def main(): 
    momentPath = "../moments_data/full_survey_moments/"
    #dir_list = filenameCleaner(os.listdir(path))
    dir_list = os.listdir(momentPath)


    # '''''
    median_df = pd.DataFrame(columns = ['filename','area','aspect','eccentricity','elongation'])

    for i in range(len(dir_list)):
        file = pd.read_csv(momentPath + dir_list[i])
        #Cleaning out NA values
        file = file.dropna()
        file = file.reset_index(drop = True)

        ### ONLY LOOKING AT FIRST 55 MIN OF FRAMES, assuming 30 fps = 99,000 frames
        file = file[file["frame"] <= (55*60*30)]

        #Adding metric medians
        newRow = pd.DataFrame() 
        newRow['filename'] = [dir_list[i]]
        newRow['area'] = [np.median(file['seg_area'])]
        newRow['aspect'] = [np.median(file['aspect_w/l'])]
        newRow['eccentricity'] = [np.median(file['eccentricity'])]
        newRow['elongation'] = [np.median(file['elongation'])]
        #newRow['area/aspect'] = [np.median(file['normal_area/aspect'])]
        #newRow['area*aspect'] = [np.median(file['normal_area*aspect'])]

        median_df = pd.concat([median_df, newRow], ignore_index = True)
        print("done with vid " + str(i))


    #print(median_df.head())

    print("\nDone calculating moment medians")

    savename = "median_moments_fullstrainsurvey"
    median_df.to_csv(savename + ".csv")
    print('Done exporting moment medians to csv')
    # '''''


    ### Data cleaning/arranging, requires median df to already exist
    # '''''
    #Reading moments csv
    momentPath = "median_moments_fullstrainsurvey.csv"
    median_df = pd.read_csv(momentPath)
    median_df.rename(columns = {'filename':'basename'}, inplace = True)
    print("\nInit median shape: " + str(median_df.shape))

    #Read in strain survey 
    strainPath = "../StrainSurveyMetaList_2019-04-09.tsv"
    strain_survey = pd.read_csv(strainPath, sep = '\t')
    print("Init strain shape: " + str(strain_survey.shape))

    #Cleaning out rows with NA weights
    strain_survey = strain_survey[strain_survey['Weight'].notna()]
    strain_survey = strain_survey.reset_index(drop = True)

    #Fixing names
    strain_survey['basename']  = [x.split('/')[-1] for x in strain_survey['NetworkFilename'].values]
    strain_survey['basename']  = [x.replace(".avi","") for x in strain_survey['basename'].values]
    median_df['basename']  = [x.replace(".avi_moments.csv","") for x in median_df['basename'].values]
    median_df = median_df.drop(['Unnamed: 0'], axis = 1) 

    print("\n")

    #Putting strain and median together
    combinedData = pd.merge(strain_survey, median_df, how = 'outer', on = 'basename')

    #Cleaning out NA values again (Missing moments data)
    combinedData = combinedData[combinedData['area'].notna()]
    combinedData = combinedData[combinedData['aspect'].notna()]
    combinedData = combinedData[combinedData['eccentricity'].notna()]
    combinedData = combinedData[combinedData['elongation'].notna()]
    combinedData = combinedData.reset_index(drop = True)

    print("\nFinal median shape: " + str(median_df.shape))
    print("Final strain shape: " + str(strain_survey.shape))
    print("Combined shape: " + str(combinedData.shape))

    combinedData['areaxeccen'] = combinedData['area']*combinedData['eccentricity']
    #print(combinedData.head())


    # combinedData['area/aspect'] = combinedData['area']/combinedData['aspect']
    # combinedData['area*aspect'] = combinedData['area']*combinedData['aspect']
    #combinedData['area*elong'] = combinedData['area']*combinedData['elongation']

    combinedData.to_csv("fullsurvey_momentmedians.csv")
    print("Appended moment median data to strain survey and saved as csv")

    #print(combinedData.isnull().sum())
    #print(combinedData[['NetworkFilename','Weight','area','aspect','eccentricity','elongation']])
    # print(combinedData['area'].sort_values())

    
    # '''''
    '''''

    
    ### Regression

    #combinedDataPath = "final_normalized_medianMetric_strainSurvey.csv"
    # combinedData = pd.read_csv(combinedDataPath)

    Y = combinedData['Weight']

    X_1 = combinedData[['area','aspect']]
    X_2 = combinedData[['area','elongation']]
    X_3 = combinedData[['area','eccentricity']]
    X_4 = combinedData[['area','aspect', 'elongation']]
    X_5 = combinedData[['area','aspect', 'eccentricity']]
    X_6 = combinedData[['area','aspect', 'elongation', 'eccentricity']]

    
    X_area = combinedData['area']   

    X_asp_mult = combinedData['area*aspect']
    X_asp_div = combinedData['area/aspect']
    X_eccen_mult = combinedData['area*eccen']
    X_elong_mult = combinedData['area*elong']
    

    #2d area vs weight
    # stat_df = pd.DataFrame([X_area,Y]).T
    # reg_area = stats.linregress(stat_df)

    stat_df = pd.DataFrame([X_asp_mult, Y]).T
    reg_asp_mult = stats.linregress(stat_df)

    stat_df = pd.DataFrame([X_asp_div, Y]).T
    reg_asp_div = stats.linregress(stat_df)  

    stat_df = pd.DataFrame([X_eccen_mult, Y]).T
    reg_eccen_mult = stats.linregress(stat_df)  

    stat_df = pd.DataFrame([X_elong_mult, Y]).T
    reg_elong_mult = stats.linregress(stat_df)  


    #Multivar
    # reg_1 = linear_model.LinearRegression()    
    # reg_2 = linear_model.LinearRegression()    
    # reg_3 = linear_model.LinearRegression()    
    # reg_4 = linear_model.LinearRegression()    
    # reg_5 = linear_model.LinearRegression()    
    # reg_6 = linear_model.LinearRegression()

    # reg_1.fit(X_1, Y) 
    # reg_2.fit(X_2, Y) 
    # reg_3.fit(X_3, Y)   
    # reg_4.fit(X_4, Y) 
    # reg_5.fit(X_5, Y) 
    # reg_6.fit(X_6, Y) 

    # print("\nArea:")
    # print("R^2: " + str(round(reg_area.rvalue**2, 4)))
    # print("slope: " + str(round(reg_area.slope, 4)))
    # print("intercept: " + str(round(reg_area.intercept, 4)))

    print("\nArea*aspect:")
    print("R^2: " + str(round(reg_asp_mult.rvalue**2, 4)))
    print("slope: " + str(round(reg_asp_mult.slope, 4)))
    print("intercept: " + str(round(reg_asp_mult.intercept, 4)))

    
    print("\nArea/aspect:")
    print("R^2: " + str(round(reg_asp_div.rvalue**2, 4)))
    print("slope: " + str(round(reg_asp_div.slope, 4)))
    print("intercept: " + str(round(reg_asp_div.intercept, 4)))

    print("\nArea*eccen:")
    print("R^2: " + str(round(reg_eccen_mult.rvalue**2, 4)))
    print("slope: " + str(round(reg_eccen_mult.slope, 4)))
    print("intercept: " + str(round(reg_eccen_mult.intercept, 4)))

    
    print("\nArea*elong:")
    print("R^2: " + str(round(reg_elong_mult.rvalue**2, 4)))
    print("slope: " + str(round(reg_elong_mult.slope, 4)))
    print("intercept: " + str(round(reg_elong_mult.intercept, 4)))

    '''''


    # plt.scatter(X_eccen_mult, Y, s=0.5)
    # plt.xlabel("Area x Eccentricity")
    # plt.ylabel("Weight")

    # linear_model = np.polyfit(X_eccen_mult, Y, 1)
    # linear_model_fn = np.poly1d(linear_model)
    # x_s=np.arange(0,1400)
    # plt.plot(x_s,linear_model_fn(x_s),color="red",  label=(round(reg_asp_mult.rvalue**2, 4)))

    # plt.show()



# r_squared = r2_score(x, y)

# fig, ax = plt.subplots() 
# ax.plot(x, y) 
# ax.set(xlabel='year', ylabel='P', title='rain') 
# ax.grid() 

# pylab.plot(x,p(x), "y--") 
# pl.plot(x, y, 'og-', label=("y=%.6fx+(%.6f) - $R^2$=%.6f"%(z[0],z[1], r_squared))) 
# pl.legend()



    # print("\nArea, aspect:")
    # print("R^2: " + str(round(reg_1.score(X_1, Y), 4)))
    # print("eqn coeffs: " + str(np.round(reg_1.coef_, 4)))
    # print("eqn intercept: " + str(round(reg_1.intercept_, 4)))

    # print("\nArea, elongation:")
    # print("R^2: " + str(round(reg_2.score(X_2, Y), 4)))
    # print("eqn coeffs: " + str(np.round(reg_2.coef_, 4)))
    # print("eqn intercept: " + str(round(reg_2.intercept_, 4)))

    # print("\nArea, eccentricity:")
    # print("R^2: " + str(round(reg_3.score(X_3, Y), 4)))
    # print("eqn coeffs: " + str(np.round(reg_3.coef_, 4)))
    # print("eqn intercept: " + str(round(reg_3.intercept_, 4)))

    # print("\nArea, aspect, elongation:")
    # print("R^2: " + str(round(reg_4.score(X_4, Y), 4)))
    # print("eqn coeffs: " + str(np.round(reg_4.coef_, 4)))
    # print("eqn intercept: " + str(round(reg_4.intercept_, 4)))

    # print("\nArea, aspect, eccentricity:")
    # print("R^2: " + str(round(reg_5.score(X_5, Y), 4)))
    # print("eqn coeffs: " + str(np.round(reg_5.coef_, 4)))
    # print("eqn intercept: " + str(round(reg_5.intercept_, 4)))

    # print("\nArea, aspect, elongation, eccentricity:")
    # print("R^2: " + str(round(reg_6.score(X_6, Y), 4)))
    # print("eqn coeffs: " + str(np.round(reg_6.coef_, 4)))
    # print("eqn intercept: " + str(round(reg_6.intercept_, 4)))

    # print("\n")
    # print("Data dimensions: " + str(combinedData.shape))
    # print("\n")


    #'''''





    # Creating the figure

    # fig = plt.figure()
    
    # ax = plt.axes(projection="3d")
 
    # # creating a wide range of points x,y,z
    # x = combinedData['area']
    # y = combinedData['aspect']
    # z = combinedData['Weight']
    
    # # plotting a 3D line graph with X-coordinate,
    # # Y-coordinate and Z-coordinate respectively
    # #ax.plot3D(x, y, z, 'red')
    
    # # plotting a scatter plot with X-coordinate,
    # # Y-coordinate and Z-coordinate respectively
    # # and defining the points color as cividis
    # # and defining c as z which basically is a
    # # defination of 2D array in which rows are RGB
    # #or RGBA
    # ax.scatter3D(x, y, z, s=0.5)

    # eq = reg.coef_[0]*x + reg.coef_[1]*y + reg.intercept_

    # ax.plot_surface(x, y, eq, 'red')
    
    # # Showing the above plot
    # plt.show()









# stat_df = np.zeros((6,4))

# # Read CSV's and create 3d stat_df
# for file in os.listdir():
#     if file.endswith(".csv"):
#         with open(file, 'r') as f:
#             sample = np.array(pd.read_csv(file))

#             stat_df = np.dstack([stat_df, sample])

# stat_df = np.delete(stat_df, 0, axis=2)
# mean_df = np.copy(stat_df[:,:,1])


# for i in range(1, stat_df.shape[0]):
#     for j in range(1, stat_df.shape[1]):
#         mean_df[i][j] = np.mean(np.abs(stat_df[i,j,:]))


# r_sq_stdev = np.std(stat_df[:,3,:], axis=1, dtype = np.float64)

# mean_df = pd.DataFrame(mean_df)
# mean_df.columns = ['metric', 'slope', 'int', "R^2"]
# mean_df["R^2 SD"] = r_sq_stdev

# # Mean
# print("\nHighest mean slope:")
# print(mean_df.sort_values(by=['slope'], ascending=False))
# print("\nHighest mean R^2:")
# print(mean_df.sort_values(by=['R^2'], ascending=False))
# print("\n")



#Median
# median_df = pd.DataFrame(median_df)
# median_df.columns = ['metric', 'slope', 'int', "R^2"]
# print("\nHighest median slope:")
# print(median_df.sort_values(by=['slope'], ascending=False))
# print("\nHighest median R^2:")
# print(median_df.sort_values(by=['R^2'], ascending=False))


if __name__ == "__main__":
    main()