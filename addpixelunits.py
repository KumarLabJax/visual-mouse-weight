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
import re
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
    parser.add_argument(
        '--in-path',
        type=str,
        default='../../fullSurvey/code/fullsurvey_momentmedians.csv',
        help='input strain survey dataframe file'
    )
    parser.add_argument(
        '--out-path',
        type=str,
        default='survey_with_corners.csv',
        help='output strain survey dataframe file'
    )
    parser.add_argument(
        '--corner-folder',
        type=str,
        default='yaml_coords/',
        help='Folder containing yaml corner files to add pixel measurements'
    )

    args = parser.parse_args()

    print("\n")

    # Load survey dataframe to append corner conversions to
    survey_path = args.in_path
    survey_df = pd.read_csv(survey_path)
    suffix_to_remove = '\.avi_moments_table1_circrect\.csv'
    survey_df['basename'] = [re.sub(suffix_to_remove, '', x) for x in survey_df['basename']]

    # Make filename & px conversion pandas df 
    conversion_df = pd.DataFrame(columns=["basename","px_conversion"])

    # Get corner files
    corner_path = args.corner_folder
    dir_list = [x for x in os.listdir(corner_path) if os.path.splitext(x)[1] == '.yaml']

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

    merged_data.to_csv(args.out_path)
    print("\nSaved merged data as 'survey_with_corners.csv'\n")



if __name__ == '__main__':
    main()

