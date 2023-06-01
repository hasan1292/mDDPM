import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm 
import argparse
import os
import sys
from pathlib import Path
import ants

def arg_parser():
    parser = argparse.ArgumentParser(
        description='Resample 3D volumes in NIfTI format')
    parser.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed')
    parser.add_argument('-modal', '--modality', type=str, required=False, default='_t1',
                    help='t1, t2, FLAIR, ...')
    parser.add_argument('-o', '--out-dir', type=str, required=False, default='tmp',
                        help='output directory for preprocessed files')
    parser.add_argument('-r', '--resolution', type=float, required=False, nargs=3, default=[1.0, 1.0, 1.0],
                        help='target resolution')
    parser.add_argument('-or', '--orientation', type=str, required=False, default='RAS',
                        help='target orientation')
    parser.add_argument('-inter', '--interpolation', type=int, required=False, default=4,
                        help='target orientation')
    parser.add_argument('-nomask', '--nomaskandseg', type=int, required=False, default=0,
                        help='set to one if you can reuse the masks and segmentations of other modalities')
    parser.add_argument('-trans', '--transform', type=str, required=False, default='Rigid',
                        help='specify the transformation')
    parser.add_argument('-templ', '--template', type=str, required=True,
                        help='path to template')
    return parser

def main(args=None):
    args = arg_parser().parse_args(args)
    src_basepath = args.img_dir #
    dest_basepath =  args.out_dir #


    if not os.path.isdir(args.img_dir):
        raise ValueError('(-i / --img-dir) argument needs to be a directory of NIfTI images.')

    Path(args.out_dir).mkdir(parents=True,exist_ok=True)

    fixed_im = ants.image_read(args.template)
    fixed_im = fixed_im.reorient_image2('RAI')

    if os.path.isdir(os.path.join(Path(args.img_dir).parents[0],'seg')) and args.nomaskandseg !=1:
        seg_path = os.path.join(Path(args.img_dir).parents[0],'seg')
        seg_out = os.path.join(Path(args.out_dir).parents[0],'seg')
        Path(seg_out).mkdir(parents=True,exist_ok=True)
    else: 
        seg_path = None

    if  os.path.isdir(os.path.join(Path(args.img_dir).parents[0],'mask')) and args.nomaskandseg !=1:
        mask_path = os.path.join(Path(args.img_dir).parents[0],'mask')
        mask_out = os.path.join(Path(args.out_dir).parents[0],'mask')
        Path(mask_out).mkdir(parents=True,exist_ok=True)
    else: 
        mask_path = None

    for i, file in tqdm(enumerate(os.listdir(args.img_dir))):
        if not os.path.isfile(os.path.join(dest_basepath, file)) or not os.path.isfile(os.path.join(mask_out, file.replace(args.modality,'_mask'))) or not os.path.isfile(os.path.join(seg_out, file.replace(args.modality,'_seg'))):
            path_img = os.path.join(args.img_dir, file)
            moving_im = ants.image_read(path_img)
            # register to template
            im_tx = ants.registration(fixed=fixed_im, moving=moving_im, type_of_transform = args.transform )
            moved_im = im_tx['warpedmovout']
            ants.image_write(moved_im, os.path.join(dest_basepath, file))

            if mask_path is not None:
                path_mask = os.path.join(mask_path, file.replace(args.modality,'_mask'))
                moving_mask = ants.image_read(path_mask)
                # transform the mask in the same way as the volume
                moved_mask = ants.apply_transforms( fixed=fixed_im, moving=moving_mask,
                                           transformlist=im_tx['fwdtransforms'] )
                ants.image_write(moved_mask, os.path.join(mask_out, file.replace(args.modality,'_mask')))

            if seg_path is not None:
                path_seg = os.path.join(seg_path, file.replace(args.modality,'_seg'))
                moving_seg = ants.image_read(path_seg)
                moving_seg = moving_seg.reorient_image2('RAI') # reorient to standard orientation
                # transform the segmentation in the same way as the volume
                moved_seg = ants.apply_transforms( fixed=fixed_im, moving=moving_seg,
                                           transformlist=im_tx['fwdtransforms'] )
                ants.image_write(moved_seg, os.path.join(seg_out, file.replace(args.modality,'_seg')))

        




if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


