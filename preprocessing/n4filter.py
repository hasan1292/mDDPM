import ants
import sys
import argparse
import os
from pathlib import Path
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser(
        description='Reorient 3D volumes in NIfTI format to RAS')
    parser.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed')
    parser.add_argument('-o', '--out-dir', type=str, required=False, default='tmp',
                        help='output directory for preprocessed files')
    parser.add_argument('-m', '--mask-dir', type=str, required=False, default=None,
                        help='mask directory for preprocessed files')
    return parser


def main(args=None):
    args = arg_parser().parse_args(args)
    # try:
    if not os.path.isdir(args.img_dir):
        raise ValueError('(-i / --img-dir) argument needs to be a directory of NIfTI images.')
    # Create output dir if it does not exist yet
    Path(args.out_dir).mkdir(parents=True,exist_ok=True)
    # Define n4 filter options
    n4_opts = {'iters': [200, 200, 200, 200], 'tol': 0.0005}
    # Iterate through input directory
    
    for i, file in enumerate(tqdm(os.listdir(args.img_dir))):
        if not os.path.isfile(os.path.join(args.out_dir, file)):
            # print('Processing file {} of {}'.format(i + 1, len(os.listdir(args.img_dir))))
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                # Define file path
                file_path = os.path.join(args.img_dir, file)
                # Read img and mask
                img = ants.image_read(file_path)
                if args.mask_dir is not None:
                    try:
                        mask = ants.image_read(os.path.join(args.mask_dir, file.replace('_t1ce', '_mask').replace('_t2', '_mask').replace('_t1', '_mask').replace('_flair', '_mask').replace('_FLAIR', '_mask').replace('_dwi', '_mask')))
                        # Smooth mask
                        smoothed_mask = ants.smooth_image(mask, 1)
                    except: 
                        smoothed_mask = None
                        print('no mask')
                    # Perform bias field correction
                    img = ants.n4_bias_field_correction(img, convergence=n4_opts, weight_mask=smoothed_mask)
                else:
                    img = ants.n4_bias_field_correction(img, convergence=n4_opts)
                # Write output img

                ants.image_write(img, os.path.join(args.out_dir,file))
                if args.out_dir == 'tmp':
                    os.remove(os.path.join(args.img_dir, file))
                    os.rename(os.path.join(args.out_dir, file), os.path.join(args.img_dir, file))
    # Delete tmo directory
    if args.out_dir == 'tmp':
        os.rmdir(args.out_dir)
    return 0
    # except Exception as e:
    #     print(e)
    #     return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
