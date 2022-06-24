export DIR_ROOT="/data/data" ## Change to your root directory
export DIR_AK_VG="$DIR_ROOT/Animal_Kingdom/video_grounding"
export DIR_VSL="$DIR_AK_VG/code/vsl"

# Create soft link to annotations and images
mkdir -p $DIR_VSL/data/dataset/ak/video
ln -sf $DIR_AK_VG/dataset/* $DIR_VSL/data/dataset/ak/video
ln -sf $DIR_AK_VG/annotation/* $DIR_VSL/data/dataset/ak
mkdir -p $DIR_VSL/data/features/ak
ln -sf $DIR_AK_VG/features/* $DIR_VSL/data/features/ak/new


# Create / Replace configuration files or codes
# mv $DIR_VSL/util/data_gen.py $DIR_VSL/util/data_gen_original.py
cp -f $DIR_AK_VG/code/code_new/vsl/util/data_gen.py $DIR_VSL/util/data_gen.py
mkdir -p $DIR_VSL/data/features/ak/new
ln -sf $DIR_AK_VG/code/code_new/vsl/data/features/ak/new/feature_shapes.json $DIR_VSL/data/features/ak/new/feature_shapes.json
# unzip $DIR_AK_VG/code/code_new/vsl/data/features/glove.840B.300d.zip
ln -sf $DIR_AK_VG/code/code_new/vsl/data/features/glove.840B.300d.txt $DIR_VSL/data/features/glove.840B.300d.txt

# Print completed process
echo "VSL Video Grounding dataset prepared"