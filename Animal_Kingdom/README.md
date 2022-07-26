# Animal Kingdom Dataset

# Instructions for using the Animal Kingdom Dataset

## Reading README files
The README files can be opened using any text editor. Alternatively, rename .md to .txt to view the contents.

## Dataset and Code
Download the dataset and code from <https://sutdcv.github.io/Animal-Kingdom>

### File size of dataset (~80GB in total)
* `action_recognition/dataset/video.tar.gz` 
    * 15.6GB --> 15.7GB uncompressed
* `action_recognition/dataset/image.tar.gz` 
    * 42.1GB --> 43.5GB uncompressed
* `action_recognition/annotation`
    * 271.6 MB

* `pose_estimation/dataset/dataset.tar.gz` 
    * 2.3GB --> 2.3GB uncompressed
* `pose_estimation/annotation` 
    * 259.3 MB

* `video_grounding/dataset.tar.gz` 
    * 15.1GB --> 15.2GB uncompressed
* `video_grounding/features.tar.gz` 
    * 1.1GB --> 1.3GB uncompressed
* `video_grounding/annotation` 
    * 1.1 MB

### Structure of folder
* Read the respective README_$TASK.MD files to set up the respective environment to run the baseline codes
* The respective codes are obtained from the original source.
    * Modifications of respective codes can be found in respective `$TASK/code_new` folders
* Helper scripts ({task}/code/code_new/prepare_dir_{task}.sh) are provided to automatically set up the environment to directly run our dataset. Please read the respective {task}/README_{task}.md files
