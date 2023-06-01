import sys
import argparse
import os
from pathlib import Path


def arg_parser():
    parser = argparse.ArgumentParser(
        description='Extract mask files from directory')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--img-dir', type=str, required=True,
                          help='path to directory with images to be processed')
    required.add_argument('-o', '--out-dir', type=str, required=True,
                          help='output directory for preprocessed files')
    return parser


def main(args=None):
    args = arg_parser().parse_args(args)
    try:
        if not os.path.isdir(args.img_dir):
            raise ValueError('(-i / --img-dir) argument needs to be a directory of NIfTI images.')
        Path(args.out_dir).mkdir(exist_ok=True)
        for file in os.listdir(args.img_dir):
            if file.endswith('_mask.nii.gz') or file.endswith('_mask.nii'):
                os.rename(os.path.join(args.img_dir, file), os.path.join(args.out_dir, file))
        return 0
    except Exception as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


