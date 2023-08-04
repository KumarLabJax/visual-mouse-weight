#!/usr/bin/env python3

"""
@author: Malachy Guzman

This file collects/analyzes/adds pre-existing corner information and appends to survey.
It's required that corner detection has already been run, 

IMPORTANT: Filepaths may need to be edited to work for external users.
"""

import argparse
import numpy as np
import pandas as pd
import os
from pathlib import Path, WindowsPath
import yaml

CORNERS_SUFFIX = '_corners_v2.yaml'
ARENA_SIZE_CM = 52

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--arena-size-cm',
        type=float,
        default=52,
        help='the arena size is used to derive cm/pixel using corners files',
    )
   
    # parser.add_argument(
    #     'corner_file',
    #     help='Corner file used to add pixel measurements'
    # )

    args = parser.parse_args()

    print("\n")

    # Load survey dataframe to append corner conversions to
    # THIS PATH WILL NEED TO BE EDITED BY EXTERNAL USERS
    survey_path = "../../fullSurvey/code/fullsurvey_momentmedians.csv"
    survey_df = pd.read_csv(survey_path)

    # Make filename & px conversion pandas df 
    conversion_df = pd.DataFrame(columns=["basename","px_conversion"])

    # Get corner files
    corner_path = "yaml_coords/"
    dir_list = os.listdir(corner_path)


    for i in range(len(dir_list)):
        with open((corner_path + dir_list[i])) as corners_file:
            corners_dict = yaml.safe_load(corners_file)
            
            xs = corners_dict['corner_coords']['xs']
            ys = corners_dict['corner_coords']['ys']

            # Get all non-diagonal pixel distances between corners and take the meadian
            xy_ul, xy_ll, xy_ur, xy_lr = [
                np.array(xy, dtype=np.float32) for xy in zip(xs, ys)
            ]
            med_corner_dist_px = np.median([
                np.linalg.norm(xy_ul - xy_ll),
                np.linalg.norm(xy_ll - xy_lr),
                np.linalg.norm(xy_lr - xy_ur),
                np.linalg.norm(xy_ur - xy_ul),
            ])

            ### Checking it all works 
            # print("\n")
            # print("left side length: " + str(np.linalg.norm(xy_ul - xy_ll)))
            # print("bottom side length: " + str(np.linalg.norm(xy_ll - xy_lr)))
            # print("right side length: " + str(np.linalg.norm(xy_lr - xy_ur)))
            # print("top side length: " + str(np.linalg.norm(xy_ur - xy_ul)))
            # print("median corner distance calculaed as: " + str(med_corner_dist_px))
            # print("Real arena side length: " + str(args.arena_size_cm) + " cm")
            # print("I calculate scaling to be: " + str(np.float32(args.arena_size_cm / med_corner_dist_px)))
            # print("\n")

            # Calculate scaling factor
            cm_per_pixel = np.float32(args.arena_size_cm / med_corner_dist_px)
            # print("pixel conversion for " + str(args.corner_file) + " is " + str(cm_per_pixel))
            
            #Adding conversion
            newRow = pd.DataFrame()
            newRow['basename'] = [dir_list[i].split("_corners_v2.yaml")[0]]
            newRow['px_conversion'] = [cm_per_pixel]
            conversion_df = pd.concat([conversion_df, newRow], ignore_index = True)


    print(str(len(dir_list)) + " files in corner folder")
    print("Survey df has dim: " + str(survey_df.shape))
    print("Conversion df has dim: " + str(conversion_df.shape))

    # Merge old survey with scaling factors
    merged_data = pd.merge(survey_df, conversion_df, how = 'left', on = 'basename')

    # Scale area by (cm/px)^2
    merged_data["norm area"] = merged_data["area"] * np.square(merged_data["px_conversion"])

    # Apply metrics over norm area
    merged_data["norm_eccen_area"] = merged_data["norm area"] * merged_data["eccentricity"]
    merged_data["norm_aspect_area"] = merged_data["norm area"] * merged_data["aspect"]
    merged_data["norm_elong_area"] = merged_data["norm area"] * merged_data["elongation"]

    print("New merged df has dim: " + str(merged_data.shape))

    merged_data.to_csv("survey_with_corners.csv")
    print("\nSaved merged data as 'survey_with_corners.csv'\n")



if __name__ == '__main__':
    main()

