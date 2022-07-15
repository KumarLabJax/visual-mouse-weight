#!/usr/bin/env python3

"""
Created 6/28/2022

@author: Malachy Guzman
"""

import ReadNPYAppend as r
import numpy as np
import math as m
import imageio
import sys
import csv

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
    
    def calc_avg_pixel_count(self):
        self.pixel_counts = []
        i = 0
        while i < self.num_of_frames:
            axis_a = 0.5 * self.data[i,2] 
            axis_b = 0.5 * self.data[i,3]
            area = m.pi * axis_a * axis_b
            if (area < 4000) and (area > 500):
                self.pixel_counts.append(area)
            i = i+1
        self.avg_pixel_count = sum(self.pixel_counts)/len(self.pixel_counts)
    
    def calc_median_pixel_count(self):
        self.pixel_counts = []
        i = 0
        while i < self.num_of_frames:
            axis_a = 0.5 * self.data[i,2] 
            axis_b = 0.5 * self.data[i,3]
            area = m.pi * axis_a * axis_b
            if (area < 4000) and (area > 500):
                self.pixel_counts.append(area)
            i = i+1
        self.pixel_counts.sort()
        mid_index = len(self.pixel_counts)//2
        self.median_pixel_count = (self.pixel_counts[mid_index] + self.pixel_counts[~mid_index])/2
    
    def calc_stretched_pixel_count(self):
        self.pixel_counts = []
        i = 0
        sorted_data = np.sort(self.data[:,3])
        q3 = np.percentile(sorted_data, 75)
        while i < self.num_of_frames:
            if self.data[i,3] >= q3:
                axis_a = 0.5 * self.data[i,2] 
                axis_b = 0.5 * self.data[i,3]
                area = m.pi * axis_a * axis_b
                if (area < 4000) and (area > 500):
                    self.pixel_counts.append(area)
            i = i+1
        self.stretched_pixel_count = sum(self.pixel_counts)/len(self.pixel_counts)
    
    def calc_scrunched_pixel_count(self):
        self.pixel_counts = []
        i = 0
        sorted_data = np.sort(self.data[:,3])
        q1 = np.percentile(sorted_data, 25)
        while i < self.num_of_frames:
            if self.data[i,3] <= q1:
                axis_a = 0.5 * self.data[i,2] 
                axis_b = 0.5 * self.data[i,3]
                area = m.pi * axis_a * axis_b
                if (area < 4000) and (area > 500):
                    self.pixel_counts.append(area)
            i = i+1
        self.scrunched_pixel_count = sum(self.pixel_counts)/len(self.pixel_counts)
        
    def calc_avg_median_pixel_count(self):
        self.pixel_counts = []
        i = 0
        sorted_data = np.sort(self.data[:,3])
        q1 = np.percentile(sorted_data, 25)
        q3 = np.percentile(sorted_data, 75)
        while i < self.num_of_frames:
            if self.data[i,3] >= q1 and self.data[i,3] <= q3:
                axis_a = 0.5 * self.data[i,2] 
                axis_b = 0.5 * self.data[i,3]
                area = m.pi * axis_a * axis_b
                if (area < 4000) and (area > 500):
                    self.pixel_counts.append(area)
            i = i+1
        self.avg_median_pixel_count = sum(self.pixel_counts)/len(self.pixel_counts)

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
        self.pixel_counts.sort()
        mid_index = len(self.pixel_counts)//2
        self.median_pixel_count = (self.pixel_counts[mid_index] + self.pixel_counts[~mid_index])/2
    
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


def main():
        filename = sys.argv[1]
        print("Loaded " + filename)

        eva = EllfitVideoAnlaysis()
        eva.extract_ellipse_data(sys.argv[2])
        print("Done extracting ellipse-fit data")

        # Gradient function. data[0] is time series of x position, [1] is y pos time series. 
        # Calculates using indices as implicit time steps. 2nd arg is polyn. order for estimation (sort of?)
        x_pos = np.array(eva.data[:,0], dtype=float)
        y_pos = np.array(eva.data[:,1], dtype=float)

        x_grad = np.gradient(x_pos, 1)
        y_grad = np.gradient(y_pos, 1)

        print("Done calculating gradients")

        # CSV creation 
        header = ["v_x", "v_y", "v_mag"]
        rows = [[],[],[]]
 
        for i in range(len(x_grad)):   
           rows.append([x_grad[i], y_grad[i], np.linalg.norm([x_grad[i], y_grad[i]])])

        csvfilename = filename + "_velocity.csv"
        with open(csvfilename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)


if __name__ == "__main__":
    main()