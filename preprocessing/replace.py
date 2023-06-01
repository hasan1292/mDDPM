import nibabel as nib
import sys
import argparse
import os


def arg_parser():
    parser = argparse.ArgumentParser(
        description='Replace string from 3D volumes file names in NIfTI format')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--img-dir', type=str, required=True,
                          help='path to directory with images to be processed')
    required.add_argument('-s', '--string', type=str, required=True, nargs=2,
                          help='string to replaced')
    return parser


def main(args=None):
    args = arg_parser().parse_args(args)
    try:
        if not os.path.isdir(args.img_dir):
            raise ValueError('(-i / --img-dir) argument needs to be a directory of NIfTI images.')
        for file in os.listdir(args.img_dir):
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                os.rename(os.path.join(args.img_dir, file), os.path.join(args.img_dir, file).replace(args.string[0].lstrip(), args.string[1].lstrip()))
        return 0
    except Exception as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


