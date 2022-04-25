#!/usr/bin/env python3

"""
Created on Thurs Feb 3 09:46:44 2022

@author: dixonk
"""

import ReadNPYAppend as r
import numpy as np
import math as m
import imageio
import sys
import csv
import matplotlib.pyplot as plt

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

    def calc_avg_pixel_count_raw(self):
        self.pixel_counts = []
        i = 0
        while i < self.num_of_frames:
            axis_a = 0.5 * self.data[i,2] 
            axis_b = 0.5 * self.data[i,3]
            area = m.pi * axis_a * axis_b
            self.pixel_counts.append(area)
            i = i+1
        self.avg_pixel_count = sum(self.pixel_counts)/len(self.pixel_counts)
    
    def calc_avg_pixel_count(self):
        self.pixel_counts = []
        i = 0
        while i < self.num_of_frames:
            axis_a = 0.5 * self.data[i,2] 
            axis_b = 0.5 * self.data[i,3]
            area = m.pi * axis_a * axis_b
            if (area < 4000) and (area > 500):
                self.pixel_counts.append(area)
                if (area > 1300) and (area < 1400):
                    print("Frame for 1300-1400 pixels: " + str(i))
                if (area > 1600) and (area < 1700):
                    print("Frame for 1600-1700 pixels: " + str(i))
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

    def distibution_of_pixel_count(self):
        with open("ellfit_distribution.csv", 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["PixelCountPerFrame"])
            for pc in self.pixel_counts_list:
                writer.writerow([pc])

    def single_frame_count(self, num):
        axis_a = 0.5 * self.data[num,2] 
        axis_b = 0.5 * self.data[num,3]
        area = m.pi * axis_a * axis_b
        return area

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

    def extract_segmentation_data_raw(self, filename):
        self.pixel_counts = []
        self.total_pixel_counts = 0
        reader = imageio.get_reader(filename)
        for frame_num, frame in enumerate(reader):
            pixel_count = np.sum(frame >= 127)
            pixel_count = pixel_count/3
            self.pixel_counts.append(pixel_count)
            self.total_pixel_counts = self.total_pixel_counts + pixel_count
        self.num_of_frames = len(self.pixel_counts)

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
                if (pixel_count > 1300) and (pixel_count < 1400):
                    print("Frame for 1300-1400 pixels: " + str(frame_num))
                if (pixel_count > 1600) and (pixel_count < 1700):
                    print("Frame for 1600-1700 pixels: " + str(frame_num))
        self.num_of_frames = len(self.pixel_counts)
    
    '''
    def extract_segmentation_data_new(self, filename, threshold, graphname):
        i = 0
        j = 0
        self.pixel_counts = []
        self.total_pixel_counts = 0
        self.num_of_frames = 0
        self.avg_pixel_count = 0
        reader = imageio.get_reader(filename)
        for frame_num, frame in enumerate(reader):
                ##starts pixel analysis at 1min mark 
                if (frame_num >= 1800):
                
                    ##options for pixel count (less strict or more strict)
                    #pixel_count = np.sum(frame >= 240)
                    #pixel_count = np.sum(frame == 255)
                    #pixel_count = np.sum(frame >= 253)
                    if (threshold == 255):
                        pixel_count = np.sum(frame == threshold)
                        pixel_count = pixel_count/3
                    else:
                        pixel_count = np.sum(frame >= threshold)
                        pixel_count = pixel_count/3
                    self.pixel_counts.append(pixel_count)
                
                    self.total_pixel_counts = self.total_pixel_counts + pixel_count

                    
                    #for single frame testing
                    if (pixel_count > 108 and pixel_count < 216 and i == 0):
                        print("Frame for 108-216 pixels: " + str(frame_num + 1800))
                        print("Pixel Count: " + str(pixel_count))
                        i = i + 1

                    if (pixel_count > 864 and pixel_count < 1080 and j == 0):
                        print("Frame for 864-1080 pixels: " + str(frame_num + 1800))
                        print("Pixel Count: " + str(pixel_count))
                        j = j + 1
                    
                    #prints numpy array for frame at 1min mark
                    #if frame_num == 3600:
                    #    np.set_printoptions(threshold=np.inf)
                    #    print(frame)
                    #    print(frame.shape)
                    #    print(np.sum(frame >= threshold))
                    
        #self.num_of_frames = frame_num
        ##makes sure correct total number of frames
        ##if pixel analysis starts at 1min mark 
        self.num_of_frames = frame_num - 1800
        self.avg_pixel_count = self.total_pixel_counts/self.num_of_frames
        
        #for single frame testing
        smallest_frame = self.pixel_counts.index(min(self.pixel_counts))
        largest_frame = self.pixel_counts.index(max(self.pixel_counts))
        print("Frame for smallest pixel count: " + str(smallest_frame+1800))
        print("Smallest pixel count: " + str(self.pixel_counts[smallest_frame]))
        print("Frame for largest pixel count: " + str(largest_frame+1800))
        print("Largest pixel count: " + str(self.pixel_counts[largest_frame]))
        
        #for distribution graphs
        counts, bins, patches = plt.hist(self.pixel_counts, bins=25, range=(0, 2700), facecolor='b')
        plt.title('Average Pixel Count: ' + str(self.avg_pixel_count))
        plt.xlabel('Pixel Count (per frame)')
        plt.xticks (bins, rotation=45)
        plt.ylabel('Frequency')
        plt.ylim(0, 150000)
        plt.savefig('/home/dixonk/results/' + graphname + '.png')
    '''
    
    def distibution_of_pixel_count(self):
        with open("segmentation_distribution.csv", 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["PixelCountPerFrame"])
            for pc in self.pixel_counts:
                writer.writerow([pc])

    def single_frame_count(self, nums, threshold, filename):
        pixel_count = []
        reader = imageio.get_reader(filename)
        for frame_num, frame in enumerate(reader):
            for number in nums:
                if (frame_num == number):
                    for t in threshold:
                        pixel_count.append((np.sum(frame >= t))/3)
        print(pixel_count)

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


def main(EVA = True, SVA = True):
    eva = EllfitVideoAnlaysis()
    sva = SegVideoAnalysis()

    print("Ellipse Method")
    eva.extract_ellipse_data('/home/dixonk/results/ellfit_data/LL1-1_BTBR_ellfit.npz')
    eva.calc_avg_pixel_count()

    print("Segmentation Method")
    sva.extract_segmentation_data('/home/dixonk/results/segmentation_videos/LL1-1_BTBR_seg.avi')

    '''
    eva = EllfitVideoAnlaysis()
    sva = SegVideoAnalysis()

    sva.extract_segmentation_data('/home/dixonk/results/segmentation_videos/LL1-1_BTBR_seg.avi')
    pc_seg_BTBR = sva.get_pixel_counts()

    sva.extract_segmentation_data('/home/dixonk/results/segmentation_videos/LL1-4_000656-F-MP13-9-42430-4-S145_seg.avi')
    pc_seg_LL1_4_S145 = sva.get_pixel_counts()

    eva.extract_ellipse_data('/home/dixonk/results/ellfit_data/LL1-1_BTBR_ellfit.npz')
    eva.calc_avg_pixel_count()
    pc_ellfit_BTBR = eva.get_pixel_counts()

    eva.extract_ellipse_data('/home/dixonk/results/ellfit_data/LL1-4_000656-F-MP13-9-42430-4-S145_ellfit.npz')
    eva.calc_avg_pixel_count()
    pc_ellfit_LL1_4_S145 = eva.get_pixel_counts()

    with open("BTBR_LL1_4_S145.csv", 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["PixelCountPerFrame", "Method", "Video"])
            for pc in pc_seg_BTBR:
                writer.writerow([pc, "Seg", "BTBR"])
            for pc in pc_seg_LL1_4_S145:
                writer.writerow([pc, "Seg", "LL1_4_S145"])
            for pc in pc_ellfit_BTBR:
                writer.writerow([pc, "Ellfit", "BTBR"])
            for pc in pc_ellfit_LL1_4_S145:
                writer.writerow([pc, "Ellfit", "LL1_4_S145"])
    '''

    '''
    eva = EllfitVideoAnlaysis()
    sva = SegVideoAnalysis()
    
    eva.extract_ellipse_data('/home/dixonk/results/ellfit_data/LL1-1_BalbcJ_ellfit.npz')
    eva.calc_avg_pixel_count_raw()
    pc_raw_ellfit_BalbcJ = eva.get_pixel_counts()
    counts, bins, patches = plt.hist(pc_raw_ellfit_BalbcJ, bins=25, range=(0, 5000), facecolor='r')
    plt.xlabel('Pixel Count (per frame)')
    plt.xticks (bins, rotation=45)
    plt.ylabel('Frequency')
    plt.ylim(0, 20000)
    plt.savefig('dist_raw_ellfit_BalbcJ')
  
    
    eva.calc_avg_pixel_count()
    pc_ellfit_BalbcJ = eva.get_pixel_counts()
    counts, bins, patches = plt.hist(pc_ellfit_BalbcJ, bins=25, range=(0, 5000), facecolor='r')
    plt.xlabel('Pixel Count (per frame)')
    plt.xticks (bins, rotation=45)
    plt.ylabel('Frequency')
    plt.ylim(0, 20000)
    plt.savefig('dist_ellfit_BalbcJ')

    sva.extract_segmentation_data_raw('/home/dixonk/results/segmentation_videos/LL1-1_BalbcJ_seg.avi')
    pc_raw_seg_BalbcJ = sva.get_pixel_counts()
    counts, bins, patches = plt.hist(pc_raw_seg_BalbcJ, bins=25, range=(0, 5000), facecolor='r')
    plt.xlabel('Pixel Count (per frame)')
    plt.xticks (bins, rotation=45)
    plt.ylabel('Frequency')
    plt.ylim(0, 20000)
    plt.savefig('dist_raw_seg_BalbcJ')

    sva.extract_segmentation_data('/home/dixonk/results/segmentation_videos/LL1-1_BalbcJ_seg.avi')
    pc_seg_BalbcJ = sva.get_pixel_counts()
    counts, bins, patches = plt.hist(pc_seg_BalbcJ, bins=25, range=(0, 5000), facecolor='r')
    plt.xlabel('Pixel Count (per frame)')
    plt.xticks (bins, rotation=45)
    plt.ylabel('Frequency')
    plt.ylim(0, 20000)
    plt.savefig('dist_seg_BalbcJ')

    eva.extract_ellipse_data('/home/dixonk/results/ellfit_data/LL1-1_BTBR_ellfit.npz')
    eva.calc_avg_pixel_count_raw()
    pc_raw_ellfit_BTBR = eva.get_pixel_counts()
    counts, bins, patches = plt.hist(pc_raw_ellfit_BTBR, bins=25, range=(0, 5000), facecolor='r')
    plt.xlabel('Pixel Count (per frame)')
    plt.xticks (bins, rotation=45)
    plt.ylabel('Frequency')
    plt.ylim(0, 20000)
    plt.savefig('dist_raw_ellfit_BTBR')

    eva.calc_avg_pixel_count()
    pc_ellfit_BTBR = eva.get_pixel_counts()
    counts, bins, patches = plt.hist(pc_ellfit_BTBR, bins=25, range=(0, 5000), facecolor='r')
    plt.xlabel('Pixel Count (per frame)')
    plt.xticks (bins, rotation=45)
    plt.ylabel('Frequency')
    plt.ylim(0, 20000)
    plt.savefig('dist_ellfit_BTBR')

    sva.extract_segmentation_data_raw('/home/dixonk/results/segmentation_videos/LL1-1_BTBR_seg.avi')
    pc_raw_seg_BTBR = sva.get_pixel_counts()
    counts, bins, patches = plt.hist(pc_raw_seg_BTBR, bins=25, range=(0, 5000), facecolor='r')
    plt.xlabel('Pixel Count (per frame)')
    plt.xticks (bins, rotation=45)
    plt.ylabel('Frequency')
    plt.ylim(0, 20000)
    plt.savefig('dist_raw_seg_BTBR')

    sva.extract_segmentation_data('/home/dixonk/results/segmentation_videos/LL1-1_BTBR_seg.avi')
    pc_seg_BTBR = sva.get_pixel_counts()
    counts, bins, patches = plt.hist(pc_seg_BTBR, bins=25, range=(0, 5000), facecolor='r')
    plt.xlabel('Pixel Count (per frame)')
    plt.xticks (bins, rotation=45)
    plt.ylabel('Frequency')
    plt.ylim(0, 20000)
    plt.savefig('dist_seg_BTBR')

    eva.extract_ellipse_data('/home/dixonk/results/ellfit_data/LL1-3_000689-F-AX5-10-42416-3-S135_ellfit.npz')
    eva.calc_avg_pixel_count_raw()
    pc_raw_ellfit_LL1_3_S135 = eva.get_pixel_counts()
    counts, bins, patches = plt.hist(pc_raw_ellfit_LL1_3_S135, bins=25, range=(0, 5000), facecolor='r')
    plt.xlabel('Pixel Count (per frame)')
    plt.xticks (bins, rotation=45)
    plt.ylabel('Frequency')
    plt.ylim(0, 20000)
    plt.savefig('dist_raw_ellfit_LL1_3_S135')

    eva.calc_avg_pixel_count()
    pc_ellfit_LL1_3_S135 = eva.get_pixel_counts()
    counts, bins, patches = plt.hist(pc_ellfit_LL1_3_S135, bins=25, range=(0, 5000), facecolor='r')
    plt.xlabel('Pixel Count (per frame)')
    plt.xticks (bins, rotation=45)
    plt.ylabel('Frequency')
    plt.ylim(0, 20000)
    plt.savefig('dist_ellfit_LL1_3_S135')

    sva.extract_segmentation_data_raw('/home/dixonk/results/segmentation_videos/LL1-3_000689-F-AX5-10-42416-3-S135_seg.avi')
    pc_raw_seg_LL1_3_S135 = sva.get_pixel_counts()
    counts, bins, patches = plt.hist(pc_raw_seg_LL1_3_S135, bins=25, range=(0, 5000), facecolor='r')
    plt.xlabel('Pixel Count (per frame)')
    plt.xticks (bins, rotation=45)
    plt.ylabel('Frequency')
    plt.ylim(0, 20000)
    plt.savefig('dist_raw_seg_LL1_3_S135')

    sva.extract_segmentation_data('/home/dixonk/results/segmentation_videos/LL1-3_000689-F-AX5-10-42416-3-S135_seg.avi')
    pc_seg_LL1_3_S135 = sva.get_pixel_counts()
    counts, bins, patches = plt.hist(pc_seg_LL1_3_S135, bins=25, range=(0, 5000), facecolor='r')
    plt.xlabel('Pixel Count (per frame)')
    plt.xticks (bins, rotation=45)
    plt.ylabel('Frequency')
    plt.ylim(0, 20000)
    plt.savefig('dist_seg_LL1_3_S135')
    
    with open("ellfit_seg_distribution.csv", 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["PixelCountPerFrame", "Method", "Video"])
            for pc in pc_raw_ellfit_BalbcJ:
                writer.writerow([pc, "Ellfit_Raw", "BalbcJ"])
            for pc in pc_ellfit_BalbcJ:
                writer.writerow([pc, "Ellfit", "BalbcJ"])
            for pc in pc_raw_seg_BalbcJ:
                writer.writerow([pc, "Seg_Raw", "BalbcJ"])
            for pc in pc_seg_BalbcJ:
                writer.writerow([pc, "Seg", "BalbcJ"])
            for pc in pc_raw_ellfit_BTBR:
                writer.writerow([pc, "Ellfit_Raw", "BTBR"])
            for pc in pc_ellfit_BTBR:
                writer.writerow([pc, "Ellfit", "BTBR"])
            for pc in pc_raw_seg_BTBR:
                writer.writerow([pc, "Seg_Raw", "BTBR"])
            for pc in pc_seg_BTBR:
                writer.writerow([pc, "Seg", "BTBR"])
            for pc in pc_raw_ellfit_LL1_3_S135:
                writer.writerow([pc, "Ellfit_Raw", "LL1_3_S135"])
            for pc in pc_ellfit_LL1_3_S135:
                writer.writerow([pc, "Ellfit", "LL1_3_S135"])
            for pc in pc_raw_seg_LL1_3_S135:
                writer.writerow([pc, "Seg_Raw", "LL1_3_S135"])
            for pc in pc_seg_LL1_3_S135:
                writer.writerow([pc, "Seg", "LL1_3_S135"])    
    '''

    if EVA == True:
        eva = EllfitVideoAnlaysis()
        '''
        #for pixel counts of single frames
        eva = EllfitVideoAnlaysis()
        eva.extract_ellipse_data('/home/dixonk/results/ellfit_data/LL1-1_BTBR_ellfit.npz')
        frame_1800 = eva.single_frame_count(1800)
        frame_3600 = eva.single_frame_count(3600)
        print("Ellipse-Fit: " + str(frame_1800) + " " + str(frame_3600) + "\n")
        '''

        '''
        #for distibution of pixels for original ellfit
        eva.extract_ellipse_data('/home/dixonk/results/ellfit_data/LL1-1_BTBR_ellfit.npz')
        eva.calc_avg_pixel_count()
        eva.distibution_of_pixel_count()
        '''
        
        eva.extract_filenames("/home/dixonk/results/ellfit_data/ellfit_data_filenames.txt")
        filenames = eva.get_file_list()
        ellfit_pixel_counts = []
        ellfit_dict = {}
        i = 0
        while i < len(filenames):
            print(filenames[i])
            eva.extract_ellipse_data(filenames[i])
            eva.calc_avg_pixel_count()
            eva.calc_median_pixel_count()
            eva.calc_stretched_pixel_count()
            eva.calc_scrunched_pixel_count()
            eva.calc_avg_median_pixel_count()
            ellfit_dict["Filename"] = filenames[i]
            ellfit_dict["Average Pixel Count"] = str(eva.get_avg_pixel_count())
            ellfit_dict["Median Pixel Count"] = str(eva.get_median_pixel_count())
            ellfit_dict["Stretched Pixel Count"] = str(eva.get_stretched_pixel_count())
            ellfit_dict["Scrunched Pixel Count"] = str(eva.get_scrunched_pixel_count())
            ellfit_dict["Average Median Pixel Count"] = str(eva.get_avg_median_pixel_count())
            ellfit_pixel_counts.append(ellfit_dict)
            ellfit_dict = {}
            i = i+1
        with open("ellfit_results.csv", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, ["Filename", "Average Pixel Count", "Median Pixel Count", "Stretched Pixel Count", "Scrunched Pixel Count", "Average Median Pixel Count"])
            writer.writeheader()
            writer.writerows(ellfit_pixel_counts)
       
    if SVA == True:
        sva = SegVideoAnalysis()
        
        '''
        #for distribution of pixel counts per frame for different thresholds for what counts as white pixel
        #sva.extract_segmentation_data_new('/home/dixonk/results/segmentation_videos/LL1-1_BTBR_seg.avi', 240, 'LL1-1_BTBR_seg.avi_240')
        sva.extract_filenames("/home/dixonk/results/segmentation_videos/seg_data_filenames.txt")
        filenames = sva.get_file_list()
        for filename in filenames:
            print(filename[40:-1])
            print("FOR 240 THRESHOLD")
            sva.extract_segmentation_data_new(filename, 240, (filename[40:-1]) + '_240')
            print("FOR 253 THRESHOLD")
            sva.extract_segmentation_data_new(filename, 253, (filename[40:-1] + '_253'))
            print("FOR 255 THRESHOLD")
            sva.extract_segmentation_data_new(filename, 255, (filename[40:-1] + '_255'))
            print("FOR 127 THRESHOLD")
            sva.extract_segmentation_data_new(filename, 127, (filename[40:-1] + '_127'))
        '''
        
        '''
        #for pixel counts of single frames
        sva = SegVideoAnalysis()
        frame_1800 = sva.single_frame_count(1800, '/home/dixonk/results/segmentation_videos/LL1-1_BTBR_seg.avi')
        frame_3600 = sva.single_frame_count(3600, '/home/dixonk/results/segmentation_videos/LL1-1_BTBR_seg.avi')
        print("Segmentation: " + str(frame_1800) + " " + str(frame_3600) + "\n")
        '''

        '''
        #for distibution of pixels for original segmentation
        sva.extract_segmentation_data('/home/dixonk/results/segmentation_videos/LL1-1_BTBR_seg.avi')
        sva.calc_avg_pixel_count()
        sva.distibution_of_pixel_count()
        '''
        
        sva.extract_filenames("/home/dixonk/results/segmentation_videos/seg_data_filenames.txt")
        filenames = sva.get_file_list()
        seg_pixel_counts = []
        seg_dict = {}
        i = 0
        while i < len(filenames):
            print(filenames[i])
            sva.extract_segmentation_data(filenames[i])
            sva.calc_avg_pixel_count()
            sva.calc_median_pixel_count()
            seg_dict["Filename"] = filenames[i]
            seg_dict["Average Pixel Count"] = str(sva.get_avg_pixel_count())
            seg_dict["Median Pixel Count"] = str(sva.get_median_pixel_count())
            seg_pixel_counts.append(seg_dict)
            seg_dict = {}
            i = i+1
        with open("seg_results.csv", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, ["Filename", "Average Pixel Count", "Median Pixel Count"])
            writer.writeheader()
            writer.writerows(seg_pixel_counts)
        

main(False, False)