#!/usr/bin/env python3

"""
Created on Tues Feb 15 09:41:41 2021

@author: dixonk
"""

import ReadNPYAppend as r
import numpy as np
import imageio
import cv2
import math as m 

def extract_ellfit_data (filedir):
    data = r.read_data(filedir)
    return data
    
def overlay (videodir, videoname, data):
    reader = imageio.get_reader(videodir)
    writer = imageio.get_writer("/home/dixonk/results/ellfit_videos/" + videoname + '_render.avi', fps=30, codec='mpeg4', quality=10)
    for i, im in enumerate(reader):
        new_frame = np.copy(im)
        axis_a = 0.5 * data[i,2] 
        axis_b = 0.5 * data[i,3]
        area = m.pi * axis_a * axis_b
        area = str(area)
        cv2.putText(new_frame, "Pixel Count: " + area, (5,35), cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(0,0,200))
        writer.append_data(new_frame.astype('u1'))
    reader.close()
    writer.close()

def main():
    data = extract_ellfit_data("/home/dixonk/results/ellfit_data/LL1-1_BalbcJ_ellfit.npz.npy")
    overlay ("/home/dixonk/results/ellfit_videos/LL1-1_BalbcJ_ellfit.avi", "LL1-1_BalbcJ_ellfit", data)

main()