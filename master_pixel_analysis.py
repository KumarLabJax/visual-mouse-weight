#!/usr/bin/env python3

"""
@author: Malachy Guzman, Kayla Dixon

This code extracts segmentation and ellipse-fit data and computes the various metrics used for prediction. 
The two classes "EllfitVideoAnlaysis" and "SegVideoAnalysis" were copied or adapted from Kayla Dixon's code. 

Returns .csv file of metric values for each frame of a given video, so each video gets its own .csv.
For a large batch of videos, this file should be run through MasterPixelAnalysis.sh.

IMPORTANT: Filepaths may need to be edited to work for external users.
"""

import ReadNPYAppend as r
import numpy as np
import pandas as pd
import imageio
import cv2
import sys


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
    
    def get_avg_pixel_count(self):
        return self.avg_pixel_count
        
    def get_median_pixel_count(self):
        return self.median_pixel_count
    
    def get_stretched_pixel_count(self):
        return self.stretched_pixel_count
    
    def get_scrunched_pixel_count(self):
        return self.scrunched_pixel_count
    
    def get_avg_median_pixel_count(self):
        return self.avg_median_pixel_count

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
            moments = cv2.moments(masked_full_frame)
            contours, hierarchy = cv2.findContours(np.uint8(masked_full_frame), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) < 1:
                #If no contours exist, fill with 0's
                moments = {'m00': 0, 'm10': 0, 'm01': 0, 'm20': 0, 'm11': 0, 'm02': 0, 'm30': 0, 'm21': 0, 'm12': 0, 'm03': 0, 'mu20': 0, 'mu11': 0, 'mu02': 0, 'mu30': 0, 'mu21': 0, 'mu12': 0, 'mu03': 0, 'nu20': 0, 'nu11': 0, 'nu02': 0, 'nu30': 0, 'nu21': 0, 'nu12': 0, 'nu03': 0}
                perimeter = 0
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



def main(): 
        filename = sys.argv[1]
        print("Loaded " + filename)

        full_df = pd.DataFrame()

        eva = EllfitVideoAnlaysis()
        eva.extract_ellipse_data(sys.argv[2])
        print("Done extracting ellipse-fit data")

        sva = SegVideoAnalysis()
        # Creates dataframe of moments
        moment_df = sva.extract_and_process_seg_data(sys.argv[3])
        print("Done extracting segmentation data and calculating moments")

        # Pulls x and y positions from ellipse data
        moment_df['x_pos'] = np.array(eva.data[:,0], dtype=float)
        moment_df['y_pos'] = np.array(eva.data[:,1], dtype=float)

        # Gradient function. data[0] is time series of x position, [1] is y pos time series. 
        # Calculates using indices as implicit time steps. 2nd arg is polyn. order for estimation (sort of?)    
        moment_df['v_x'] = np.gradient(moment_df['x_pos'], 1)
        moment_df['v_y'] = np.gradient(moment_df['y_pos'], 1)
        moment_df['v_mag'] = np.linalg.norm(moment_df[['v_x','v_y']].values, axis=1)
        print("Done calculating gradients")

        # Getting the rest of the fields for csv. Theta through elongation code below taken from Brian Geuther's MouseSleep code (2021)
        moment_df['x_mom'] = moment_df['m10']/moment_df['m00']
        moment_df['y_mom'] = moment_df['m01']/moment_df['m00']
        moment_df['a'] = moment_df['m20']/moment_df['m00']-moment_df['x_mom']**2
        moment_df['b'] = 2*(moment_df['m11']/moment_df['m00'] - moment_df['x_mom']*moment_df['y_mom'])
        moment_df['c'] = moment_df['m02']/moment_df['m00'] - moment_df['y_mom']**2
        moment_df['w'] = np.sqrt(8*(moment_df['a']+moment_df['c']-np.sqrt(moment_df['b']**2+(moment_df['a']-moment_df['c'])**2)))/2
        moment_df['l'] = np.sqrt(8*(moment_df['a']+moment_df['c']+np.sqrt(moment_df['b']**2+(moment_df['a']-moment_df['c'])**2)))/2
        #moment_df['theta'] = 1/2.*np.arctan(2*moment_df['b']/(moment_df['a']-moment_df['c']))
        moment_df['aspect_w/l'] = moment_df['w']/moment_df['l']
        moment_df['circularity'] = moment_df['m00']*4*np.pi/moment_df['perimeter']**2
        moment_df['rectangular'] = moment_df['m00']/(moment_df['w']*moment_df['l'])
        moment_df['eccentricity'] = np.sqrt(moment_df['w']**2 + moment_df['l']**2)/moment_df['l']
        moment_df['elongation'] = (moment_df['mu20'] + moment_df['mu02'] + (4 * moment_df['mu11']**2 + (moment_df['mu20'] - moment_df['mu02'])**2)**0.5) / (moment_df['mu20'] + moment_df['mu02'] - (4 * moment_df['mu11']**2 + (moment_df['mu20'] - moment_df['mu02'])**2)**0.5)
        moment_df['area_x_eccen'] = moment_df['seg_area'] * moment_df['eccentricity']
        print("Done calculating moments")

        # Reorders columns and drops unnecesary info
        full_df = moment_df.reindex(['frame', 'x_pos', 'y_pos', 'v_x', 'v_y', 'v_mag', 'seg_area', 'm00', 'x_mom', 'y_mom', 'aspect_w/l','eccentricity', 'elongation', 'circularity', 'rectangular'], axis = 'columns')
        print('Done reordering columns')

        # THIS PATH MAY NEED TO BE EDITED BY EXTERNAL USERS
        # Write dataframe to csv 
        full_df.to_csv(filename.replace("code/","") + '_moments_table1_circrect.csv', index = False)
        print('Done: exported video analysis to csv')

       


if __name__ == "__main__":
    main()