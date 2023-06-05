# Visual Mouse Mass Documentation and Notes
## Malachy Guzman
## June 4, 2023
### This file explains how to use the various scripts for the visual mouse mass prediction pipeline.
### Includes important unfinished tasks at the bottom.
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

#
## To do
There are a few outstanding tasks (as of 6/4/2023):

1. If we want to include circularity and rectangularity in Table 1 of the paper, someone will have to run the prediction pipeline from step 5 onward. A few days ago I modified and reran `master_pixel_analysis.py` to include circ and rect, so there's now a new set of individual moment csv's labeled as `*_moments_table1_circrect.csv`. You could follow the exact same pipeline to produce a copy of the strain survey with moment data that includes circ and rect, and from there follow the steps in the R file to report the performance
2. On this same point of circ and rect, Table 1 in the manuscript is incomplete. The most recent idea (with Brian) is to build 6 versions of the full model, each using a different geometric variable and one with no geom var. The point is to demonstrate that eccentricity is the best metric of the ones we looked at. To report this, you would run the 50-fold crossvalidation for table 1 starting on line 452 of the R file. After getting the data together and importing to R, you would run lines `454-551` at once, and then run `554-555` to print one row of the table at a time. Change `k` on `554` to be whatever row you want to print. You would also have to redefine the variables in the `lm` models on `498-503`, see the csv or infer from the R code for the proper variable names. The geometric variables by model would be: Eccentricity, Aspect Ratio, Elongation, Circularity, Rectangularity, and None (swap each of these out)
3. The supplement hasn't been filled in on the manuscript yet. I've prepared one dataset for it, which is the strain-wise breakdown of performance and error metrics. This is finished and just has to be reported on a table. It's saved in the R and data section of my files as `strainwise_means_SDs_obs_preds.csv`. 
4. ***Most Important Task:*** We (Gautam and I) discovered a typo a couple days ago in the definition of eccentricity that pervades the whole thing. On line `198` of `master_pixel_analysis.py`, eccentricity is defined as $(\sqrt{l^2+w^2})/l$, where $w$ and $l$ are the minor and major axis of the ellipse. However, the standard definition of eccentricity is $(\sqrt{l^2-w^2})/l$ or $\sqrt{1-w^2/l^2}$. Thus our eccentricity measure actually differs by a minus sign everywhere. This likely has a significant impact on what the models do. It's not clear what happens if we correct eccentricity and rerun everything, as I haven't had time to get to that. **Confer with Brian, all this stuff was carried over from his work on the sleep code, so he knows best about this (and generally).** It isn't necessarily bad, because what we calculate is a geometric variable dependent on the ellipse just like standard eccentricity, but if we decide to keep it and not report standard eccentricity, we must rename it, otherwise it will be very confusing. 
   
## On the paper:
1. Last time we discussed we're shooting for *Cell Patterns*. 
2. The manuscript: There's one edit from Brian's round of revisions on the paper I haven't gotten to yet, it deals with the descriptions of the models M1-M4 and their performance. Methods and statistical analysis haven't been revised. Gautam took a brief look at the statistical analysis and thought it was generally good, and commented that it isn't necessary to demonstrate some of the fine details like "Fitting automatically with a LASSO provides very similar results to our full model", saying it is enough. But definitely needs review. I loosely followed the statistical analysis section in the STAR methods of the gait paper. 
3. References: all the references on box are in `references_new.bib` on overleaf, which are printed at the end of the manuscript. However, I haven't worked any of the references Vivek sourced into the text (particularly the intro), which might be good to give more depth to what I've written on relevance, competing ideas, etc. All of the references I found have been cited parenthetically like "[1]" in the text. 
4. Acknowledgments: Haven't been done. The acknowledgments from the previous paper are still present, just commented out.  I've added Kayla Dixon's acknowledgement for the pixel counting code and preliminary analysis, but might do well with a little expanding. Probably a good idea to mention the academic year fellowship, see next point.  
5. Authors and affiliations: Current author list is Malachy, Brian, Gautam, and Vivek. I can't really think of anyone else. Obviously Brian, Gautam, and Vivek are associated with JAX. I marked myself as also affiliated with JAX since that's where I did the work, even though I am now done. That seems more appropriate than my college, though if we want to acknowledge the academic year fellowship (which should probably be done), then maybe it would be good to add both jax and Carleoton College as my affiliations, because the fellowship is joint. Technically, I did the independent study with Dr. Amy Cszimar-Dalal in CS at Carleton, but we have actually never met or spoken about anything to do with this, so I'm not sure if she should be mentioned. 
