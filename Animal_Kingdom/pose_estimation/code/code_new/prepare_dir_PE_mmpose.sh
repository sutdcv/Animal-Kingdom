export DIR_ROOT="/data/data" # Change to your root directory
export DIR_AK_PE="$DIR_ROOT/Animal_Kingdom/pose_estimation"
export DIR_MMPOSE="$DIR_AK_PE/code/mmpose"

# Create soft link to annotations and images
mkdir -p $DIR_MMPOSE/data/ak_P1/annot
mkdir -p $DIR_MMPOSE/data/ak_P1/images
ln -sf $DIR_AK_PE/annotation_coco/ak_P1/* $DIR_MMPOSE/data/ak_P1/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_MMPOSE/data/ak_P1/images

mkdir -p $DIR_MMPOSE/data/ak_P2/annot
mkdir -p $DIR_MMPOSE/data/ak_P2/images
ln -sf $DIR_AK_PE/annotation_coco/ak_P2/* $DIR_MMPOSE/data/ak_P2/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_MMPOSE/data/ak_P2/images

mkdir -p $DIR_MMPOSE/data/ak_P3_amphibian/annot
mkdir -p $DIR_MMPOSE/data/ak_P3_amphibian/images
ln -sf $DIR_AK_PE/annotation_coco/ak_P3_amphibian/* $DIR_MMPOSE/data/ak_P3_amphibian/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_MMPOSE/data/ak_P3_amphibian/images

mkdir -p $DIR_MMPOSE/data/ak_P3_bird/annot
mkdir -p $DIR_MMPOSE/data/ak_P3_bird/images
ln -sf $DIR_AK_PE/annotation_coco/ak_P3_bird/* $DIR_MMPOSE/data/ak_P3_bird/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_MMPOSE/data/ak_P3_bird/images

mkdir -p $DIR_MMPOSE/data/ak_P3_fish/annot
mkdir -p $DIR_MMPOSE/data/ak_P3_fish/images
ln -sf $DIR_AK_PE/annotation_coco/ak_P3_fish/* $DIR_MMPOSE/data/ak_P3_fish/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_MMPOSE/data/ak_P3_fish/images

mkdir -p $DIR_MMPOSE/data/ak_P3_mammal/annot
mkdir -p $DIR_MMPOSE/data/ak_P3_mammal/images
ln -sf $DIR_AK_PE/annotation_coco/ak_P3_mammal/* $DIR_MMPOSE/data/ak_P3_mammal/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_MMPOSE/data/ak_P3_mammal/images

mkdir -p $DIR_MMPOSE/data/ak_P3_reptile/annot
mkdir -p $DIR_MMPOSE/data/ak_P3_reptile/images
ln -sf $DIR_AK_PE/annotation_coco/ak_P3_reptile/* $DIR_MMPOSE/data/ak_P3_reptile/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_MMPOSE/data/ak_P3_reptile/images


# Create / Replace configuration files or codes
cp -f $DIR_AK_PE/code/code_new/mmpose/configs/_base_/datasets/ak.py $DIR_MMPOSE/configs/_base_/datasets/ak.py

mkdir -p $DIR_MMPOSE/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ak/
ln -sf $DIR_AK_PE/code/code_new/mmpose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ak/* $DIR_MMPOSE/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ak/

cp -f $DIR_AK_PE/code/code_new/mmpose/mmpose/datasets/datasets/animal/animal_ak_dataset.py $DIR_MMPOSE/mmpose/datasets/datasets/animal/animal_ak_dataset.py

# mv $DIR_MMPOSE/mmpose/datasets/__init__.py $DIR_MMPOSE/mmpose/datasets/__init__original.py
# mv $DIR_MMPOSE/mmpose/datasets/datasets/__init__.py $DIR_MMPOSE/mmpose/datasets/datasets/__init__original.py
# mv $DIR_MMPOSE/mmpose/datasets/datasets/animal/__init__.py $DIR_MMPOSE/mmpose/datasets/datasets/animal/__init__original.py
cp -f $DIR_AK_PE/code/code_new/mmpose/mmpose/datasets/__init__.py $DIR_MMPOSE/mmpose/datasets/__init__.py
cp -f $DIR_AK_PE/code/code_new/mmpose/mmpose/datasets/datasets/__init__.py $DIR_MMPOSE/mmpose/datasets/datasets/__init__.py
cp -f $DIR_AK_PE/code/code_new/mmpose/mmpose/datasets/datasets/animal/__init__.py $DIR_MMPOSE/mmpose/datasets/datasets/animal/__init__.py


# Print completed process
echo "MMPose Pose Estimation dataset prepared"