#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz
        
The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""


# python register.py --moving /home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder/Task110_RUNMC/imagesTr/Case08_0001.nii.gz --fixed /home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder/Task110_RUNMC/imagesTr/Case08_0000.nii.gz --moving_seg /home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder_BACKUP/Task110_RUNMC/labelsTr/Case08_0001.nii.gz --model /home/aranem_locale/Desktop/MICCAI_2023/experiments/vxm_torch_250_110_ncc_dice/0200.pt --out /home/aranem_locale/Desktop/MICCAI_2023/experiments/vxm_torch_250_110_ncc_dice/eval/Case08/ --gt /home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder_BACKUP/Task110_RUNMC/labelsTr/Case08_0000.nii.gz


import os
import argparse
import pystrum
import SimpleITK as sitk

# third party
import numpy as np
import torch
import shutil
# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8


def register():
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='out path')
    parser.add_argument('--gt', required=True, help='GT segmentation path')
    parser.add_argument('--moving', required=True, help='moving image (source) filename')
    parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
    parser.add_argument('--moving_seg', required=True, help='moving image (source) filename')
    parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
    parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')
    parser.add_argument('--do_rigid', action='store_true',
                        help='specify moving and fixed image should be rigidly aligned first')
    parser.add_argument('--seg', action='store_true',
                        help='Set this to use VxM U-Net as segmentation network.')
    parser.add_argument('--bic', action='store_true',
                        help='Set this to use BiC U-Net as segmentation network.')
    parser.add_argument('--ilt', action='store_true',
                        help='Set this to use ILT U-Net as segmentation network.')
    args = parser.parse_args()

    # assert args.seg and args.bic, "Please set both seg and bic for segmentation!"

    # device handling
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # load moving and fixed images
    add_feat_axis = not args.multichannel
    moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
    moving_seg = vxm.py.utils.load_volfile(args.moving_seg, add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed, _ = vxm.py.utils.load_volfile(args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

    # load and set up model
    if args.bic:
        model = vxm.BiC.BiC.load(args.model, device)
    elif args.ilt:
        model = vxm.ILT.ILT.load(args.model, device)
    elif args.seg:
        model = vxm.networks.UNet.load(args.model, device)
    else:
        model = vxm.networks.VxmDense.load(args.model, device)
    # model = vxm.networks.VxmDense.load(args.model, device) if not args.seg else vxm.networks.UNet.load(args.model, device)
    model.to(device)
    model.eval()

    # rigid alignment if desired
    if args.do_rigid and not args.seg:
        mov = sitk.GetImageFromArray(moving[0, ..., 0].transpose(2, 0, 1))
        mov_s = sitk.GetImageFromArray(moving_seg[0, ..., 0].transpose(2, 0, 1))
        fix = sitk.GetImageFromArray(fixed[0, ..., 0].transpose(2, 0, 1))
        moving, moving_seg = vxm.torch.utils.rigid_align(fix, mov, mov_s)
        # -- Transpose back to nibabel file order -- #
        moving = moving.transpose(1, 2, 0)[np.newaxis, ..., np.newaxis]
        moving_seg = moving_seg.transpose(1, 2, 0)[np.newaxis, ..., np.newaxis]

    # set up tensors and permute
    input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
    input_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)
    input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)

    # predict
    if args.seg:
        y_seg = model(input_fixed)
        y_seg = torch.nn.functional.softmax(y_seg, dim=1).float()
    else:
        moved, warp, y_seg = model(input_moving, input_fixed, input_seg, registration=True)

        # Black-White grid and warp for later plotting
        # g = pystrum.pynd.ndutils.bw_grid(input_moving.size()[2:], 1)[np.newaxis, np.newaxis, ...]
        # g = torch.from_numpy(g).to(device).float()
        # g = model.transformer(g, warp)

        # save images
        moved = moved.permute(0, 2, 3, 4, 1).detach().cpu().numpy().squeeze()#.transpose(0, 2, 1)
        # y_seg = y_seg.permute(0, 2, 3, 4, 1).detach().cpu().numpy().squeeze()#.transpose(0, 2, 1)
        # g = g.detach().cpu().numpy().squeeze()#.transpose(0, 2, 1)
        warp = warp.permute(0, 2, 3, 4, 1).detach().cpu().numpy().squeeze()
        # y_seg[y_seg != 0] = 1

        y_seg = y_seg.argmax(1).detach().cpu().numpy().squeeze().transpose(2, 1, 0).astype(float)

        shutil.copy(args.moving, os.path.join(args.out, 'moving_img.nii.gz'))
        shutil.copy(args.moving_seg, os.path.join(args.out, 'moving_seg.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(moved.transpose(2, 0, 1).swapaxes(-2,-1)[...,::-1]), os.path.join(args.out, 'moved_img.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(warp.transpose(2, 0, 1, 3).swapaxes(-3,-2)[...,::-1, :]), os.path.join(args.out, 'flow.nii.gz'))
        # np.save(os.path.join(args.out, 'flow_grid'), g)
        # sitk.WriteImage(sitk.GetImageFromArray(y_seg.transpose(2, 0, 1).swapaxes(-2,-1)[...,::-1]), os.path.join(args.out, 'moved_seg.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(y_seg), os.path.join(args.out, 'moved_seg.nii.gz'))

    shutil.copy(args.fixed, os.path.join(args.out, 'fixed_img.nii.gz' if not args.seg else 'img.nii.gz'))
    if args.gt:
        shutil.copy(args.gt, os.path.join(args.out, 'fixed_seg.nii.gz' if not args.seg else 'seg_gt.nii.gz'))

    if args.seg:
        y_seg = y_seg.argmax(1).detach().cpu().numpy().squeeze().transpose(2, 1, 0).astype(float)
        sitk.WriteImage(sitk.GetImageFromArray(y_seg), os.path.join(args.out, 'pred_seg.nii.gz'))
    
# -- Main function for setup execution -- #
def main():
    register()

if __name__ == "__main__":
    register()