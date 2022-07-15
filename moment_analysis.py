#!/usr/bin/env python3

"""
Created 7/11/2022

@author: Malachy Guzman

Ellipse and segmentation data extraction adapted from Kayla Dixon's pixel_analysis.
Code for theta, aspect ratio, circularity, rectangularity, eccentricity, and elongation 
adapted from Brian Geuther's MouseSleep code.

This code does data analysis to decide which metrics are best in which cases.
"""

import ReadNPYAppend as r
import numpy as np
import pandas as pd
import imageio
import cv2
import os
import sys, time, csv, argparse
from scipy import stats



class EllfitVideoAnlaysis:
    
    def __init__(self):
        self.data = np.empty((1,1))
        self.num_of_frames = 0
        self.avg_pixel_count = 0
        self.median_pixel_count = 0
        self.stretched_pixel_count = 0
        self.scrunched_pixel_count = 0
        self.avg_median_pixel_count = 0
        self.pixel_counts = []
        self.file_list = []
    
    def extract_filenames(self, filename):
        file = open (filename, 'r')
        line = file.readline()
        while line != "":
            words = line.split('\n')
            self.file_list.append(words[0])
            line = file.readline()
        file.close()
    
    def extract_ellipse_data(self, filename):
        self.data = r.read_data(filename)
        self.num_of_frames = self.data.shape[0]
    
    def get_num_of_frames(self):
        return self.num_of_frames

    def get_file_list(self):
        return self.file_list

    def get_pixel_counts(self):
        return self.pixel_counts

class SegVideoAnalysis:
    
    def __init__(self):
        self.num_of_frames = 0
        self.avg_pixel_count = 0
        self.median_pixel_count = 0
        self.total_pixel_counts = 0
        self.file_list = []
        self.pixel_counts = []
    
    def extract_filenames(self, filename):
        file = open (filename, 'r')
        line = file.readline()
        while line != "":
            words = line.split('\n')
            self.file_list.append(words[0])
            line = file.readline()
        file.close()

    def extract_segmentation_data(self, filename):
        self.pixel_counts = []
        self.total_pixel_counts = 0
        reader = imageio.get_reader(filename)
        for frame_num, frame in enumerate(reader):
            pixel_count = np.sum(frame >= 127)
            pixel_count = pixel_count/3
            if (pixel_count > 500) and (pixel_count < 4000):
                self.pixel_counts.append(pixel_count)
                self.total_pixel_counts = self.total_pixel_counts + pixel_count
        self.num_of_frames = len(self.pixel_counts)

    def calc_avg_pixel_count(self):
        self.avg_pixel_count = self.total_pixel_counts/self.num_of_frames

    def calc_median_pixel_count(self):
        self.median_pixel_count =  np.median(self.pixel_counts)
    
    def get_num_of_frames(self):
        return self.num_of_frames
    
    def get_avg_pixel_count(self):
        return self.avg_pixel_count

    def get_median_pixel_count(self):
        return self.median_pixel_count
    
    def get_file_list(self):
        return self.file_list

    def get_pixel_counts(self):
        return self.pixel_counts

    def extract_and_process_seg_data(self, filename):
        self.pixel_counts = []
        self.total_pixel_counts = 0
        result_df = []

        reader = imageio.get_reader(filename)

        for frame_num, frame in enumerate(reader):
            ### Calculates pixel count
            pixel_count = np.sum(frame >= 127)
            pixel_count = pixel_count/3
            if (pixel_count > 500) and (pixel_count < 4000):
                self.pixel_counts.append(pixel_count)
                self.total_pixel_counts = self.total_pixel_counts + pixel_count

            ### Calculates moments
            frame = frame[:,:,0]
            masked_full_frame = np.zeros_like(frame)
            masked_full_frame[frame > 128] = 1
            #print(masked_full_frame.shape)
            moments = cv2.moments(masked_full_frame)
            contours, hierarchy = cv2.findContours(np.uint8(masked_full_frame), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            #If no contours exist, fill with 0's
            if len(contours) < 1:
                # Default values
                moments = {'m00': 0, 'm10': 0, 'm01': 0, 'm20': 0, 'm11': 0, 'm02': 0, 'm30': 0, 'm21': 0, 'm12': 0, 'm03': 0, 'mu20': 0, 'mu11': 0, 'mu02': 0, 'mu30': 0, 'mu21': 0, 'mu12': 0, 'mu03': 0, 'nu20': 0, 'nu11': 0, 'nu02': 0, 'nu30': 0, 'nu21': 0, 'nu12': 0, 'nu03': 0}
                perimeter = 0
            #Leave moments as it was and do the rest
            else:  
                max_contour = None
                max_size = -1
                for k in contours:
                    blob_size = cv2.contourArea(k)
                    if blob_size > max_size:
                        max_contour = k
                        max_size = blob_size
                perimeter = cv2.arcLength(max_contour, True)
            
            moments['frame'] = frame_num
            moments['seg_area'] = pixel_count
            moments['perimeter'] = perimeter
            result_df.append(pd.DataFrame(moments, index=[1]))

        result_df = pd.concat(result_df).reset_index(drop=True)
        self.num_of_frames = len(self.pixel_counts)
        return result_df

    def process_video(self):
        result_df = []

        for frameIndex in np.arange(self.get_num_of_frames()):
            frame = getframe(frameIndex)
            masked_full_frame = np.zeros_like(frame)
            masked_full_frame[frame > 128] = 1
            moments = cv2.moments(masked_full_frame)
            contours, hierarchy = cv2.findContours(np.uint8(masked_full_frame), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            #If no contours exist, fill with 0's
            if len(contours) < 1:
                # Default values
                moments = {'m00': 0, 'm10': 0, 'm01': 0, 'm20': 0, 'm11': 0, 'm02': 0, 'm30': 0, 'm21': 0, 'm12': 0, 'm03': 0, 'mu20': 0, 'mu11': 0, 'mu02': 0, 'mu30': 0, 'mu21': 0, 'mu12': 0, 'mu03': 0, 'nu20': 0, 'nu11': 0, 'nu02': 0, 'nu30': 0, 'nu21': 0, 'nu12': 0, 'nu03': 0}
                perimeter = 0
            #Leave moments as it was and do the rest
            else:  
                max_contour = None
                max_size = -1
                for k in contours:
                    blob_size = cv2.contourArea(k)
                    if blob_size > max_size:
                        max_contour = k
                        max_size = blob_size
                perimeter = cv2.arcLength(max_contour, True)
                
            moments['perimeter'] = perimeter
            result_df.append(pd.DataFrame(moments, index=[1]))

        result_df = pd.concat(result_df).reset_index(drop=True)

    def binned_speed_median(self, filename, ):
            self.pixel_counts = []
            self.total_pixel_counts = 0



            result_df = []

            reader = imageio.get_reader(filename)

            for frame_num, frame in enumerate(reader):
                ### Calculates pixel count
                pixel_count = np.sum(frame >= 127)
                pixel_count = pixel_count/3

                if (pixel_count > 500) and (pixel_count < 4000):
                    #if(speed)
                    self.pixel_counts.append(pixel_count)
                    self.total_pixel_counts = self.total_pixel_counts + pixel_count

               
               
               
            #result_df = pd.concat(result_df).reset_index(drop=True)
            self.num_of_frames = len(self.pixel_counts)
            return result_df


def overall_med_metric_comparison(df, metric):
    #want to produce the line equation and R^2 value for each area vs metric plot
    x = df["seg_area"]
    y = df[metric]

    stat_df = pd.DataFrame([x,y]).T
    
    # Cleaning out na values
    stat_df = stat_df.dropna()
    stat_df = stat_df.reset_index(drop = True)

    res = stats.linregress(stat_df)
    info = [round(res.slope,4), round(res.intercept,4), round(res.rvalue**2,4)]
    
    # print("Cleaned data shape: "+str(stat_df.shape))
    #print(metric + " slope = " + str(info[0]) + ", int = "+str(info[1])+", R^2 = "+str(info[2]))

    return info




def main(): 
        #This is the moments csv, not video
        filename = sys.argv[1]
        print("\n")
        print("Loaded " + filename.replace("code/",""))

        metric_list = ["v_mag", "eccentricity", "elongation", "aspect_w/l", "circularity", "rectangular"]
        important_moment_df = pd.read_csv(filename, encoding = 'utf-8')
        comparison_df = pd.DataFrame()

        fulltime_start = time.time()

        #median area over the video with frames sampled from those with a certain speed. 4 speed bins: low, med, high, and overall
        #speed_med_bin = sva.binned_metric_median(filename, metric_list[0])      
        
        for i in metric_list:
            comparison_df = comparison_df.append(pd.DataFrame(overall_med_metric_comparison(important_moment_df, i)).T, ignore_index=True)
            
        comparison_df.columns = ['slope', 'intercept', 'r_sq']
        comparison_df.index = metric_list

        # print("\n")
        # print(comparison_df.sort_values(by=['r_sq'], ascending = False))

        # Calculates processing time
        fulltime_end = time.time()
        processtime = fulltime_end-fulltime_start #in seconds
        print("Took " + str(processtime) + " sec to do calculations.") #in s
        
        #Write dataframe to csv

        # dir = 'C://projects/kumar-lab/guzmam/fullSurvey/moments_data'
        # file_name = 'test.txt'
        # fname = os.path.join(dir, file_name)
        # file = open(fname,'w')
        savename = filename.replace("code/","")
        savename = savename.replace(".avi.csv","")

        comparison_df.to_csv(savename + "_moment_stats.csv")
        print('Done: exported moment analysis to csv')
        print("\n")
    


if __name__ == "__main__":
    main()