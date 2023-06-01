import ants
import nibabel as nib

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm 

def arg_parser():
    parser = argparse.ArgumentParser(
        description='Resample 3D volumes in NIfTI format')
    parser.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed')
    parser.add_argument('-o', '--out-dir', type=str, required=False, default='tmp',
                        help='output directory for preprocessed files')
    parser.add_argument('-r', '--resolution', type=float, required=False, nargs=3, default=[1.0, 1.0, 1.0],
                        help='target resolution')
    parser.add_argument('-or', '--orientation', type=str, required=False, default='RAI',
                        help='target orientation')
    parser.add_argument('-inter', '--interpolation', type=int, required=False, default=4,
                        help='target orientation')
    return parser


def main(args=None):
    args = arg_parser().parse_args(args)
    try:
        if not os.path.isdir(args.img_dir):
            raise ValueError('(-i / --img-dir) argument needs to be a directory of NIfTI images.')
        Path(args.out_dir).mkdir(parents=True ,exist_ok=True)
        for i, file in tqdm(enumerate(os.listdir(args.img_dir))):
            # print('Processing file {} of {}'.format(i + 1, len(os.listdir(args.img_dir))))
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                if not os.path.isfile(os.path.join(args.out_dir, file)):
                    try:
                        vol = ants.image_read(os.path.join(args.img_dir, file))
                        if args.resolution != vol.spacing:
                            vol = ants.resample_image(vol, args.resolution, False, args.interpolation)
                        vol = vol.reorient_image2(args.orientation)
                        ants.image_write(vol, os.path.join(args.out_dir, file))
                        if args.out_dir == 'tmp':
                            os.remove(os.path.join(args.img_dir, file))
                            os.rename(os.path.join(args.out_dir, file), os.path.join(args.img_dir, file))
                    except Exception as e:
                        print('An error occurred: {}'.format(str(e)))
                        print('Could not process file {}. Moving on...'.format(file))
        # Delete tmp directory
        if args.out_dir == 'tmp':
            os.rmdir(args.out_dir)
        return 0
    except Exception as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
