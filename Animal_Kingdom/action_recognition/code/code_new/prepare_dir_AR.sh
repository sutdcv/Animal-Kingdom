export DIR_ROOT="/data/data" ## Change to your root directory
export DIR_AK_AR="$DIR_ROOT/Animal_Kingdom/action_recognition"
export DIR_SLOWFAST="$DIR_AK_AR/code/slowfast"

# Create soft link to annotations and images
mkdir -p $DIR_SLOWFAST/data/ak
mkdir -p $DIR_SLOWFAST/data/ak/annot
ln -sf $DIR_AK_AR/dataset/* $DIR_SLOWFAST/data/ak
ln -sf $DIR_AK_AR/annotation/* $DIR_SLOWFAST/data/ak/annot

# Create / Replace configuration files or codes
mkdir -p $DIR_SLOWFAST/configs/AK
ln -sf $DIR_AK_AR/code/code_new/slowfast/configs/AK/* $DIR_SLOWFAST/configs/AK

export DIR_ANACONDA_ENV="/data/anaconda3/envs/slowfast"
export DIR_PYTORCHVIDEO_DIST="$DIR_ANACONDA_ENV/lib/python3.8/site-packages/pytorchvideo/layers/distributed.py"

mv $DIR_SLOWFAST/setup.py $DIR_SLOWFAST/setup_original.py 
ln -sf $DIR_AK_AR/code/code_new/slowfast/setup.py $DIR_SLOWFAST
# mv $DIR_SLOWFAST/slowfast/models/losses.py $DIR_SLOWFAST/slowfast/models/losses_original.py 
cp -f $DIR_AK_AR/code/code_new/slowfast/slowfast/models/losses.py $DIR_SLOWFAST/slowfast/models

mv $DIR_PYTORCHVIDEO_DIST $DIR_AK_AR/code/code_new/pytorchvideo/layers/distributed_original.py
ln -sf $DIR_AK_AR/code/code_new/pytorchvideo/layers/distributed.py $DIR_PYTORCHVIDEO_DIST

# Install earlier version of protobuf
python -m pip install protobuf==3.20.0

# Print completed process
echo "SlowFast Action Recognition dataset prepared"