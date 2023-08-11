# Visual Mouse Mass Documentation and Notes
### Authors: Malachy Guzman, Brian Geuther, Gautam Sabnis, and Vivek Kumar
### This readme explains how to use the various scripts for the visual mouse mass prediction pipeline, from video to prediction in grams.
### This repository contains all the files associated with the manuscript *Highly Accurate and Precise Determination of Mouse Mass Using Computer Vision*, 2023. See that paper for full details of methods, models, and use cases.


## Inference and Prediction Pipeline
1. Start with a set of open field videos 
   
2. Run `PixelAnalysis.sh`. This produces the segmentation and ellipse fit videos and the associated files. Line 40 runs an earlier version of analysis, but if you just want the seg and ell videos, get rid of it. Note `line 40` is commented out, leave as such.

3. Run `MasterPixelAnalysis.sh`. This is intended to be run with a batch file, a list of return-delimited video file names saved as a `.txt`. This shell script runs the python file `master_pixel_analysis.py` *for each video*, which performs the pixel counting, calculates the contours, moments, and metrics, compiles it into a dataframe, and exports it to `.csv`. The csv is a set of time series where the sample rate is 1 frame, i.e. as fast as possible. Note that `master_pixel_analysis.py` runs once for each video, so there is now a csv file for each video. If you just have one video, you could just run `master_pixel_analysis.py` instead of the `.sh` form.

4. Run `video_summarizer.py`. This file compresses each video's individual time series into medians and appends them to the corresponding row of the strain survey metadata file, `StrainSurveyMetaList_2019-04-09.tsv`. External users should replace this file with their own video metadata.
   1. **IMPORTANT:** See singularity note below, it is not guaranteed that this pipeline will work outside the Tracking Singularity Container. 
#
**Arena Normalization:** At this point, we can use our basic statistical models, as we have all the information except corner position. To make the conversion from pixel area $A_{px}$ to unit converted area $A_{cm}$, we need to do the following:
- Run the corner detection network adapted from HRNet, methods specified in *Sheppard, et al. Stride-level analysis of mouse open field behavior using deep-learning-based pose estimation. Cell Reports, 2022*. This network outputs `.yaml` files identifying the coordinates of the corners.
  
- To add corner normalization, run `addpixelunits.py`. This requires that the corner detection network has already been run, and that the `.yaml` files containing the corner data exist. In this final iteration, the moments csv is referred to as `survey_with_corners.csv` (see notes for more detail).
#
To add **Relative Standard Deviation (RSD)** data, run `RSD_analysis.py`. This only requires that some version of the strain survey moments csv exists, with area and eccentric area present, but can be easily modified to compute RSD of any variables. 
   1. **Note:** this file does not append to the strain survey, instead it outputs one dataframe with the RSDs for all the videos. The RSD .csv is merged with the strain survey in R.


## Statistics, Modeling, and Data
1. The file `final_modeling.R` handles all of the stats and modeling. This code takes the strain survey with moments data, the RSD data, and the individual moments data. 
2. Figure data:
   1. Individual video moment data is used in fig 1b
   2. RSD data is used in fig 2 and the supplement.
   3. The full strain survey moments data is used for figs 3 and 4 and the supplement.


## Notes
### Singularity:
- All testing was run inside the original tracking environment used in [the tracking paper codebase](https://github.com/KumarLabJax/MouseTracking), which uses python 3.5.2. We include [env.txt](env.txt) for the used libraries within that environment to report the exact library versions.

### Length of Videos (55 min)
- Most videos in our dataset are 1-2 hours long. However, they come from  assays which involve adding things into the open field that lowers the quality of the segmentation mask.
- All videos are guaranteed to be normal (i.e. no objects aside from the individual mouse in the arena, aside from an arm when the mouse is put in) for the first 55 minutes of recording.
- Therefore, ***Only report, infer, or regress on the first 55 minutes of data.*** 
  - This is very easy to handle with the direct plotting of individual video moment data, as you can just crop the time series to whatever time you like, as with fig 1b. 
  - With RSD, 55 minute cropping is built into `RSD_analysis.py`. Note that you can't crop after computing RSD's, as they're a summary statistic. 
  - The same is true for `video_summarizer.py`, where the time series have to be cropped BEFORE computing any medians or statistics. 
  - All of this is already built in. 

