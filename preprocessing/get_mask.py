import nibabel as nib
import sys
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import ants
def arg_parser():
    parser = argparse.ArgumentParser(
        description='create binary mask for 3D volumes in NIfTI format')
    parser.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed')
    parser.add_argument('-o', '--out-dir', type=str, required=False, default='tmp',
                        help='output directory for preprocessed files')
    parser.add_argument('-mod', '--modality', type=str, required=True,
                        help='output directory for preprocessed files')

    return parser


def main(args=None):
    args = arg_parser().parse_args(args)
    try:
        if not os.path.isdir(args.img_dir):
            raise ValueError('(-i / --img-dir) argument needs to be a directory of NIfTI images.')
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        for i, file in tqdm(enumerate(os.listdir(args.img_dir))):
            # print('Processing file {} of {}'.format(i + 1, len(os.listdir(args.img_dir))))
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                vol_file = ants.image_read(os.path.join(args.img_dir, file))
                mask = ants.get_mask(vol_file)
                ants.image_write(mask, os.path.join(args.out_dir,file.replace(args.modality,'mask')))
        
                if args.out_dir == 'tmp':
                    os.remove(os.path.join(args.img_dir, file))
                    os.rename(os.path.join(args.out_dir, file), os.path.join(args.img_dir, file))
        # Delete tmo directory
        if args.out_dir == 'tmp':
            os.rmdir(args.out_dir)
        return 0
    except Exception as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
