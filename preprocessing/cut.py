import nibabel as nib
import numpy as np
import sys
import argparse
import os
from pathlib import Path
from tqdm import tqdm

def first_nonzero(arr, axis, invalid_val=0):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis, invalid_val=0):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def arg_parser():
    parser = argparse.ArgumentParser(
        description='Extract mask files from directory')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--img-dir', type=str, required=True, nargs='+',
                          help='path to directory with images to be processed')
    required.add_argument('-m', '--mask-dir', type=str, required=True,
                          help='mask directory')
    required.add_argument('-o', '--output', type=str, required=True,
                          help='output directory')
    required.add_argument('-mode', '--mode', type=str, required=False, default='t1',
                        help='mode')
    return parser


def main(args=None):
    args = arg_parser().parse_args(args)
    try:
        Path(args.output).mkdir(parents=True,exist_ok=True)
        Path(args.output + '/mask/').mkdir(parents=True,exist_ok=True)
        Path(args.output + '/seg/').mkdir(parents=True,exist_ok=True)
        Path(args.output + '/' + args.mode).mkdir(parents=True,exist_ok=True)
        file_suffix = args.mode
        for input_dir in args.img_dir:
            if not os.path.isdir(input_dir):
                raise ValueError('(-i / --img-dir) argument needs to be a directory of NIfTI images.')
        
        for i, mask_path in tqdm(enumerate(os.listdir(args.mask_dir))):
            try:
                if mask_path.endswith('_mask.nii.gz') or mask_path.endswith('_mask.nii'):
                    if not os.path.isfile(args.output + '/' + file_suffix + '/' + mask_path.replace('mask', file_suffix)) or not os.path.isfile(args.output + '/seg/' + mask_path.replace('mask', 'seg')):
                        # print('Processing file {} of {}'.format(i+1, len(os.listdir(args.mask_dir))))
                        max_dims = [0, 0, 0]
                        # Load mask
                        mask_file = nib.load(os.path.join(args.mask_dir, mask_path))
                        mask = mask_file.get_fdata()

                        # Zero axis
                        zero_min_indices = first_nonzero(mask, 0, 999999).min() 
                        zero_max_indices = last_nonzero(mask, 0).max()

                        # First axis
                        first_min_indices = first_nonzero(mask, 1, 999999).min()
                        first_max_indices = last_nonzero(mask, 1).max()

                        # Second axis
                        second_min_indices = first_nonzero(mask, 2, 999999).min()
                        second_max_indices = last_nonzero(mask, 2).max()

                        max_dims = np.maximum(max_dims, [zero_max_indices - zero_min_indices,
                                                        first_max_indices - first_min_indices,
                                                        second_max_indices - second_min_indices])
                        print(max_dims)
                        # for k, input_dir in enumerate(args.img_dir):
                            
                        
                        # Create path if it does not exist yet
                        # Path(file_suffix + '-cut').mkdir(exist_ok=True)

                        # Construct new file name
                        file_name = mask_path.replace('mask', file_suffix)
                        out_name =  args.output + '/' + file_suffix + '/' + mask_path.replace('mask', file_suffix) # 
                        mask_name = args.output + '/mask/' + mask_path
                        seg_name = args.output + '/seg/' + mask_path.replace('mask', 'seg')
                        # Load volume
                        vol = nib.load(os.path.join(input_dir, file_name))
                        vol_np = vol.get_fdata()

                        # Slice
                        new_vol = vol_np[zero_min_indices:zero_max_indices, first_min_indices:first_max_indices,
                                    second_min_indices:second_max_indices]

                        # Create new nifti file and save it
                        new_vol_file = nib.Nifti1Image(new_vol, affine=vol.affine, header=vol.header)
                        nib.save(new_vol_file,out_name)

                        # Path('mask-cut').mkdir(exist_ok=True)
                        new_mask_vol = mask[zero_min_indices:zero_max_indices, first_min_indices:first_max_indices,
                                    second_min_indices:second_max_indices]
                        new_mask_file = nib.Nifti1Image(new_mask_vol, affine=mask_file.affine, header=mask_file.header)

                        if os.path.isfile((args.mask_dir+mask_path).replace('mask','seg')) and not os.path.isfile(seg_name):
                            seg_path = os.path.join(args.mask_dir, mask_path).replace('mask','seg')
                            seg = nib.load(seg_path)
                            seg_np = seg.get_fdata()

                            # Slice
                            new_seg = seg_np[zero_min_indices:zero_max_indices, first_min_indices:first_max_indices,
                                    second_min_indices:second_max_indices]
                            # Create new nifti file and save it
                            new_seg_file = nib.Nifti1Image(new_seg, affine=vol.affine, header=vol.header)
                            nib.save(new_seg_file,seg_name)
                        # print(mask_name)
                        if not os.path.isfile(mask_name):

                            nib.save(new_mask_file, mask_name)
            except: 
                print('error')
        print('Maximum dimensions are {}'.format(max_dims))
        return 0
    except Exception as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
