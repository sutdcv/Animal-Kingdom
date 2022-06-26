# Video Grounding

## Dataset and Code
* [Download dataset and code here](https://forms.office.com/r/WCtC0FRWpA)

## Structure of Video Grounding Dataset
* Annotations are stored in .txt format 
* Annotations:
    * `clip` `segment_start_time` `segment_end_time` `##` `"annnotation_sentence"`

## Evaluation Metrics
* We follow VSLNet and use both Recall@1 and Mean IoU
* For the evaluation code, please refer to <https://github.com/IsaacChanghau/VSLNet/blob/master/util/runner_utils_t7.py>


## Instructions to run Video Grounding models
1. To prepare the environment, please refer to 
    * [LGI] <https://github.com/JonghwanMun/LGI4temporalgrounding>
    * [VSLNet] <https://github.com/IsaacChanghau/VSLNet>

2. For feature extraction, we used the I3D model and frame rate of 24 fps.

3. Move and replace files according to the directories in `$DIR_AK_VG/video_grounding/code/code_new`:
    * Helper script to move / create symbolic links to files
        * Remember to change the root directory `$DIR_ROOT` in `$DIR_AK_VG/video_grounding/code/code_new/prepare_dir_VG.sh`
        * `bash $DIR_AK_VG/video_grounding/code/code_new/prepare_dir_VG.sh`
