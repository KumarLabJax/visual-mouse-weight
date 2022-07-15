#!/usr/bin/env python3

"""
Created on 6/21/22
@author: Malachy Guzman
This is a prototype for analyzing pixel count at a given rate instead of for the whole video.
Adapted from Kayla Dixon's pixel_analysis.py. 
"""

from time import time
import ReadNPYAppend as r
import numpy as np
import math as m
import imageio
import sys
import csv


class SegVideoAnalysis:
    
    def __init__(self):
        self.num_of_frames = 0
        self.avg_pixel_count = 0
        self.median_pixel_count = 0
        self.total_pixel_counts = 0
        self.file_list = []
        self.pixel_counts = []
        self.timePixels= []
    

    def extract_filenames(self, filename):
        file = open (filename, 'r')
        line = file.readline()
        while line != "":
            words = line.split('\n')
            self.file_list.append(words[0])
            line = file.readline()
        file.close()


    #Counts pixels from segmentation video
    def extract_segmentation_data(self, filename):
        self.pixel_counts = []  #Array contianing number of mouse pixels in each frame
        self.total_pixel_counts = 0
        reader = imageio.get_reader(filename)
        for frame_num, frame in enumerate(reader):
            pixel_count = np.sum(frame >= 127)  #this sums the number of pixels in a given frame w/ value>=127
            pixel_count = pixel_count/3         #still don't know why this is here
            if (pixel_count > 500) and (pixel_count < 4000):    #Threshold
                self.pixel_counts.append(pixel_count)   
                self.total_pixel_counts = self.total_pixel_counts + pixel_count #sum of all mouse pixels
        self.num_of_frames = len(self.pixel_counts)


    #Calculates a median from the pixel array. This seems unnecessary
    def calc_median_pixel_count(self):
        self.median_pixel_count =  np.median(self.pixel_counts)
    

    #Creates array of median pixel number in a given time interval
    #Should be rewritten as a map instead of a 1D array
    def segMedian_time(self, frameInterval):

        #Assuming 30fps, use this if you want time based instead of frame based
        # frameInterval = timeInterval * 30 
        self.timePixels = list()

        print('\ntotal frame #: ' + str(self.num_of_frames))
        print('median interval #: ' + str(self.num_of_frames//frameInterval))

        #Faster version
        # 0 -> frame num/10 frames/sample = number of samples  

        for i in range(self.num_of_frames//frameInterval):
            #List of frames to sample 
            intervalFrames = self.pixel_counts[(i*frameInterval):((i*frameInterval)+frameInterval)]
            self.timePixels.append(np.median(intervalFrames))

        print('successfully saved ' + str(self.num_of_frames//frameInterval) 
        + ' medians at ' + str(frameInterval) + ' frames per sample.\n')

            

    def get_num_of_frames(self):
        return self.num_of_frames

    def get_median_pixel_count(self):
        return self.median_pixel_count

    def get_segMedian_time(self, timeInterval):
        return self.segMedian_time(self, timeInterval)

    def get_timePixels(self):
        return self.timePixels
    
    def get_file_list(self):
        return self.file_list

    def get_pixel_counts(self):
        return self.pixel_counts


def main():
        filename = sys.argv[1]
        frameInterval = 15   #Sample every x frames

        sva = SegVideoAnalysis()

        sva.extract_segmentation_data(sys.argv[1])  #Only requires seg video as cmd line arg
        sva.calc_median_pixel_count()
        sva.segMedian_time(frameInterval)

        
        #csv of every frame in vid:
        # header_big = [filename.replace('../intervalTest/',''), 'frame', 'mouse pixels']
        # rows_big=[]
        
        # #Appending time data
        # for i in range(len(sva.timePixels)):   
        #     rows_big.append(['', i, str(sva.get_pixels_counts())[i]])

        # csvfilename = filename + "_timeAnalysis.csv"
        # with open(csvfilename, 'w') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(header_big)
        #     writer.writerows(rows_big)



        #Creates csv with medians
        # header_med = [filename.replace('../intervalTest/',''), 'full median pixel count', 'time interval', 'median px count in '+str(timeInterval)+'sec intervals']
        # rows_med=[]
        # rows_med.append(['', str(sva.get_median_pixel_count()), 0, str(sva.get_timePixels()[0])])



        header_med = [filename.replace('../intervalTest/','')]
        rows_med=[]
        
        #Appending time data
        for i in range(len(sva.timePixels)):   
            rows_med.append([str(sva.get_timePixels()[i])])

        csvfilename = filename + "_TimeAnalysis.csv" #i.e. B6J3_TimeAnalysis.csv
        with open(csvfilename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header_med)
            writer.writerows(rows_med)
        

if __name__ == "__main__":
    main()