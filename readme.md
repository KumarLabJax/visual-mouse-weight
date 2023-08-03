# Visual Mouse Mass Documentation and Notes
## Malachy Guzman
### This file explains how to use the various scripts for the visual mouse mass prediction pipeline.
#
## Inference and Prediction Pipeline: 
1. Start with an open field video (or many)
2. Run `PixelAnalysis.sh` with `line 40` commented out. This produces the segmentation and ellipse fit videos and the associated files. Line 40 runs an earlier version of analysis, but if you just want the seg and ell videos, get rid of it.
3. Run `MasterPixelAnalysis.sh`. This is intended to be run with a batch file, a list of return-delimited video file names saved as a `.txt`. This shell script runs the python file `master_pixel_analysis.py` *for each video*, which performs the pixel counting, calculates the contours, moments, and metrics, compiles it into a dataframe, and exports it to `.csv`. The csv is a time series where the sample rate is 1 frame, i.e. as fast as possible. Note that `master_pixel_analysis.py` runs once for each video, so there is now a csv file for each video.
4. Run `multiple_regression.py`. This file compresses each video's individual time series into medians and appends them to the corresponding row of the strain survey file, `StrainSurveyMetaList_2019-04-09.tsv`. 
   1. (Note that the naming of `multiple_regression.py` is misleading, it no longer does the regressions, that has now been moved to R. This file just performs data management. Should be renamed, but I no longer have cluster access.) 
   2. `MasterPixelAnalysis.sh` must be run on Sumner. 
   3. **IMPORTANT:** See singularity note below, it is not possible to just run any of the python files alone, they must be run inside of the UPenn Singularity Container. 
5. We now have the "Moments Data" for the whole strain survey. This is sufficient to start analyzing.
6. To add corner normalization, run `addpixelunits_MGmass.py`. This requires that the corner detection network has already been run, and that the `.yaml` files containing the corner data exist. This is a one line command saved on Confluence under the documentation section of my Academic Year folder. `addpixelunits_MGmass.py` requires step 4 to have already happened, as it appends new columns of scaled variables to the moments data csv. In this final iteration, the moments csv is referred to as `survey_with_corners.csv` (see notes for more detail).
7. To add relative standard deviation (RSD) data, run `1c_RSD_moment_analysis.py`. This is self contained and has no arguments, but requires that some version of the strain survey moments csv exists. 
   1. **Note:** this files does not append to the strain survey, instead it outputs one dataframe with the RSDs for all the videos. The RSD csv is merged with the strain survey in R
   2. This file should also be renamed, as fig 1c is now fig 2. I would actually just delete the number and call it `RSD_analysis.py` or somehting along those lines. 

#
## Statistics, Modeling, and Data
1. The file `revised_AcYr_Modeling.R` handles all of the stats and modeling, and is internally commented. Gautam has reviewed it and likes it, as of June 2, 2023.
2. The R code takes the strain survey with moments data, the RSD data, and the individual moments data. You can of course run any of these individually, they aren't interdependent. 
3. Figure data:
   1. Individual video moment data is used in fig 1b
   2. RSD data is used in fig 2
   3. the full strain survey moments data is used for figs 3 and 4.
4. That's all the data. :) 
   
#
## Notes

### Singularity:
- As mentioned above, any of the standalone python files must be run inside of the singularity container `UPennInferImg.simg`.
- To enter you must already be in an interactive session (or be performing a standalone job submit as in the `.sh` files)
- At least one place the container is located is `/projects/kumar-lab/guzmam/environment/UPennInferImg.simg`
- To enter the container, run the next two lines one after the other in an interactive session (on sumner or winter):
  - $ module load singularity
  - $ singularity shell "container path"
- You can then run whatever you like (so long as its packages are contained in the container).

### Length of Videos (55 min)
- Most videos in the strain survey are 1-2 hours long. However, they come from drug assays, some of which involve adding things into the open field that lowers the quality of the segmentation mask.
- All videos are guaranteed to be normal (i.e. no objects aside from the individual mouse in the arena, aside from an arm when the mouse is put in) for the first 55 minutes of recording.
- Therefore, ***Only report, infer, or regress on the first 55 minutes of data.*** 
  - This is very easy to handle with the direct plotting of individual video moment data, as you can just crop the time series to whatever time you like, as with fig 1b. 
  - With RSD, 55 minute cropping is built into `1c_RSD_moment_analysis.py`. Note that you can't crop after computing RSD's, as they're a summary statistic. 
  - The same is true for `multiple_regression.py`, where the time series have to be cropped BEFORE computing any medians or statistics. 
  - All of this is already built in. 

