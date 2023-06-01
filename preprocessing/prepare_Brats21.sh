#!/bin/bash
# cli arguments: 
# 1. path to data directory
# 2. path to output directory
INPUT_DIR=$1
DATA_DIR=$2

# make the arguments mandatory and that the data dir is not a relative path
if [ -z "$INPUT_DIR" ] || [ -z "$DATA_DIR" ] 
then
  echo "Usage: ./prepare_MSLUB.sh <input_dir> <output_dir>"
  exit 1
fi

if [ "$INPUT_DIR" == "." ] || [ "$INPUT_DIR" == ".." ]
then
  echo "Please use absolute paths for input_dir"
  exit 1
fi  
# For BRATS, we already have resampled, skull-stripped data 

mkdir -p $DATA_DIR/v2skullstripped/Brats21/
mkdir -p $DATA_DIR/v2skullstripped/Brats21/mask

cp -r  $INPUT_DIR/t2  $INPUT_DIR/seg $DATA_DIR/v2skullstripped/Brats21/

echo "extract masks"
python get_mask.py -i $DATA_DIR/v2skullstripped/Brats21/t2 -o $DATA_DIR/v2skullstripped/Brats21/t2 -mod t2 
python extract_masks.py -i $DATA_DIR/v2skullstripped/Brats21/t2 -o $DATA_DIR/v2skullstripped/Brats21/mask
python replace.py -i $DATA_DIR/v2skullstripped/Brats21/mask -s " _t2" ""

echo "Register t2"
python registration.py -i $DATA_DIR/v2skullstripped/Brats21/t2 -o $DATA_DIR/v3registered_non_iso/Brats21/t2 --modality=_t2 -trans Affine -templ sri_atlas/templates/T1_brain.nii


echo "Cut to brain"
python cut.py -i $DATA_DIR/v3registered_non_iso/Brats21/t2 -m $DATA_DIR/v3registered_non_iso/Brats21/mask/ -o $DATA_DIR/v3registered_non_iso_cut/Brats21/ -mode t2

echo "Bias Field Correction"
python n4filter.py -i $DATA_DIR/v3registered_non_iso_cut/Brats21/t2 -o $DATA_DIR/v4correctedN4_non_iso_cut/Brats21/t2 -m $DATA_DIR/v4correctedN4_non_iso_cut/Brats21/mask
mkdir $DATA_DIR/v4correctedN4_non_iso_cut/Brats21/mask
cp $DATA_DIR/v3registered_non_iso_cut/Brats21/mask/* $DATA_DIR/v4correctedN4_non_iso_cut/Brats21/mask
mkdir $DATA_DIR/v4correctedN4_non_iso_cut/Brats21/seg
cp $DATA_DIR/v3registered_non_iso_cut/Brats21/seg/* $DATA_DIR/v4correctedN4_non_iso_cut/Brats21/seg
echo "Done"

# now, you should copy the files in the output directory to the data directory of the project
