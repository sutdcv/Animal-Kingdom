# Pose Estimation

## Dataset and Code
* [Download dataset and code here](https://forms.office.com/r/WCtC0FRWpA)

## Structure of Pose Estimation Dataset
* Annotations follow MPII format and are stored in .json format
* Annotations:
    * `image`: Path to image

    * `animal`: Name of animal
    * `animal_parent_class`: Parent class of the animal (e.g., Amphibian)
    * `animal_class`: Class of the animal (e.g., Amphibian)
    * `animal_subclass`: Subclass of the animal (e.g., Frog / Toad)

    * `joints_vis`: Visibility of joints (1 means visible, 0 means not visible)
    * `joints`: Coordinates of the joints. All images are in 640×360 px(width × height) resolution. Invisible joint coordinates are [-1, -1].
 
    * `scale`: Scale of bounding box with respect to 200px
    * `center`: Coordinates of the centre point of the bounding box

* There are 23 keypoints in the following order:
    * `joint_id`: 
        <details><summary>Click to show list of keypoints</summary>

        * 0: Head_Mid_Top
        * 1: Eye_Left 
        * 2: Eye_Right 
        * 3: Mouth_Front_Top 
        * 4: Mouth_Back_Left
        * 5: Mouth_Back_Right
        * 6: Mouth_Front_Bottom
        * 7: Shoulder_Left
        * 8: Shoulder_Right
        * 9: Elbow_Left
        * 10: Elbow_Right
        * 11: Wrist_Left
        * 12: Wrist_Right
        * 13: Torso_Mid_Back
        * 14: Hip_Left
        * 15: Hip_Right
        * 16: Knee_Left
        * 17: Knee_Right
        * 18: Ankle_Left 
        * 19: Ankle_Right
        * 20: Tail_Top_Back
        * 21: Tail_Mid_Back
        * 22: Tail_End_Back

        </details>

## Evaluation Metric
* We chose PCK@0.05. 
* For the evaluation code, please refer to <https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/lib/core/evaluate.py>


## Instructions to run Pose Estimation models
This code was separately tested on RTX 3090, and 3080Ti using CUDA10.2.

1. To prepare the environment, refer to 
    * [HRNet] <https://github.com/leoxiaobin/deep-high-resolution-net.pytorch>
    * [HRNet-DARK] <https://github.com/ilovepose/DarkPose#distribution-aware-coordinate-representation-for-human-pose-estimation>

    * **IMPORTANT**: Perform the next step (Step 2) first before performing make in make libs (Step 4) in HRNet so that the dataset will be initialized

2. Move and replace files according to the directories in `$DIR_AK_AR/pose_estimation/code/code_new`:
    * Helper script to move / create symbolic links to files
        * Remember to change the root directory `$DIR_ROOT` in `$DIR_AK/pose_estimation/code/code_new/prepare_dir_PE.sh`
        * `bash $DIR_AK/pose_estimation/code/code_new/prepare_dir_PE.sh`

3. Untar the dataset
    * `tar -zxvf $DIR_AK/pose_estimation/dataset.tar.gz`

4. Execute the code
    * `python tools/train.py --cfg $DIR_HRNET/experiments/mpii/hrnet/w32_256x256_adam_lr1e-3_ak.yaml `

5. [Alternative] We have also specially prepared the dataset for use in MMPose <https://mmpose.readthedocs.io/en/latest/get_started.html> by OpenMMLab. 
    * COCO annotations are available (Not used in our experiments)
    * Only mAP metric is available (Not used in our experiments) for COCO datasets in MMPose <https://github.com/open-mmlab/mmpose/issues/721#issuecomment-859453118>, <https://github.com/open-mmlab/mmpose/issues/707>
    * Helper script to set up environment
        * Remember to change the root directory `$DIR_ROOT` in `$DIR_AK/pose_estimation/code/code_new/prepare_dir_PE.sh`
        * `bash $DIR_AK/pose_estimation/code/code_new/prepare_dir_PE_mmpose.sh`
        * `python $DIR_MMPOSE/tools/train.py configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ak/hrnet_w32_ak_256x256.py`


## Solutions to potential issues:
<details><summary>Click to expand</summary>

1. unable to execute 'gcc': No such file or directory. error: command 'gcc' failed with exit status 1
    * `sudo apt install gcc`

2. ModuleNotFoundError: No module named 'nms.cpu_nms'
    * <https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/24>
    * `cd $DIR_HRNET/lib`
    * `make`

3. OSError: The nvcc binary could not be located in your $PATH. Either add it to your path, or set $CUDAHOME
    * <https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/143>
    * `export CUDAHOME="/usr/lib/cuda"`

4. OSError: The CUDA nvcc path could not be located in /usr/lib/cuda/bin/nvcc
    * Ensure cuda and nvcc are installed 
        * `sudo apt install nvidia-cuda-toolkit`
        * `which nvcc`
            * should show: `/usr/bin/nvcc`
        * `echo $CUDAHOME`
            * should show: `/usr/lib/cuda`

    * `sudo ln -s /usr/bin/nvcc /usr/lib/cuda/bin/nvcc`

5. RuntimeError: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 804: forward compatibility was attempted on non supported HW
    * Driver may have been uninstalled after running `sudo apt install nvidia-cuda-toolkit`

    * Check if the driver is installed
        * `nvidia-smi`
            * should show the drivers available for installation (e.g., `sudo apt install nvidia-utils-470`)

6. AttributeError: module 'torch.onnx' has no attribute 'set_training'
    * <https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/230>
    * `pip install tensorboardX --upgrade`
    * `pip install tensorboard`

7. ImportError: libcudart.so.10.2: cannot open shared object file: No such file or directory
    * <https://itsfoss.com/solve-open-shared-object-file-quick-tip>
    * `sudo /sbin/ldconfig -v`

</details>
