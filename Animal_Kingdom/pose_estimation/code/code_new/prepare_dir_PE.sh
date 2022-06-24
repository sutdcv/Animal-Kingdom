export DIR_ROOT="/data/data" # Change to your root directory
export DIR_AK_PE="$DIR_ROOT/Animal_Kingdom/pose_estimation"
export DIR_HRNET="$DIR_AK_PE/code/hrnet"

# Create soft link to annotations and images
mkdir -p $DIR_HRNET/data/ak_P1/annot
mkdir -p $DIR_HRNET/data/ak_P1/images
ln -sf $DIR_AK_PE/annotation/ak_P1/* $DIR_HRNET/data/ak_P1/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_HRNET/data/ak_P1/images

mkdir -p $DIR_HRNET/data/ak_P2/annot
mkdir -p $DIR_HRNET/data/ak_P2/images
ln -sf $DIR_AK_PE/annotation/ak_P2/* $DIR_HRNET/data/ak_P2/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_HRNET/data/ak_P2/images

mkdir -p $DIR_HRNET/data/ak_P3_amphibian/annot
mkdir -p $DIR_HRNET/data/ak_P3_amphibian/images
ln -sf $DIR_AK_PE/annotation/ak_P3_amphibian/* $DIR_HRNET/data/ak_P3_amphibian/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_HRNET/data/ak_P3_amphibian/images

mkdir -p $DIR_HRNET/data/ak_P3_bird/annot
mkdir -p $DIR_HRNET/data/ak_P3_bird/images
ln -sf $DIR_AK_PE/annotation/ak_P3_bird/* $DIR_HRNET/data/ak_P3_bird/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_HRNET/data/ak_P3_bird/images

mkdir -p $DIR_HRNET/data/ak_P3_fish/annot
mkdir -p $DIR_HRNET/data/ak_P3_fish/images
ln -sf $DIR_AK_PE/annotation/ak_P3_fish/* $DIR_HRNET/data/ak_P3_fish/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_HRNET/data/ak_P3_fish/images

mkdir -p $DIR_HRNET/data/ak_P3_mammal/annot
mkdir -p $DIR_HRNET/data/ak_P3_mammal/images
ln -sf $DIR_AK_PE/annotation/ak_P3_mammal/* $DIR_HRNET/data/ak_P3_mammal/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_HRNET/data/ak_P3_mammal/images

mkdir -p $DIR_HRNET/data/ak_P3_reptile/annot
mkdir -p $DIR_HRNET/data/ak_P3_reptile/images
ln -sf $DIR_AK_PE/annotation/ak_P3_reptile/* $DIR_HRNET/data/ak_P3_reptile/annot
ln -sf $DIR_AK_PE/dataset/* $DIR_HRNET/data/ak_P3_reptile/images


# Create / Replace configuration files or codes
ln -sf $DIR_AK_PE/code/code_new/hrnet/experiments/mpii/hrnet/w32_256x256_adam_lr1e-3_ak.yaml $DIR_HRNET/experiments/mpii/hrnet/

ln -sf $DIR_AK_PE/code/code_new/hrnet/lib/dataset/ak.py $DIR_HRNET/lib/dataset/ak.py
mv $DIR_HRNET/lib/dataset/__init__.py mv $DIR_HRNET/lib/dataset/__init__original.py
ln -sf $DIR_AK_PE/code/code_new/hrnet/lib/dataset/__init__.py $DIR_HRNET/lib/dataset/__init__.py


# Print completed process
echo "HRNet Pose Estimation dataset prepared"