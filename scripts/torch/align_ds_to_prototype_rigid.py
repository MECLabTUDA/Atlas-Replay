import os, shutil
import numpy as np
import SimpleITK as sitk
from voxelmorph.torch.utils import rigid_align

old_task = 'Task113_UCL'
task_name = 'Task123_UCL'
dataset = '/home/aranem_locale/Desktop/MICCAI_2023/MICCAI_2023_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task113_UCL'
out_folder = dataset.replace(old_task, task_name)
prototype = '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid_pseudo_rehearsal/vxm_torch_250_110_121_112_ncc_ce/Prototype/prototype_img_updated.nii.gz'
prototype_seg = '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid_pseudo_rehearsal/vxm_torch_250_110_121_112_ncc_ce/Prototype/prototype_seg_updated.nii.gz'

if __name__ == "__main__":
    # Copy data first
    shutil.copytree(dataset, out_folder, dirs_exist_ok=True)
    if 'Decath' in out_folder:
        cases = [x.split('_000')[0] for x in os.listdir(os.path.join(out_folder, 'imagesTr')) if '._' not in x and '.json' not in x and 'DS_Store' not in x]
    else:
        cases = [x.split('_')[0] for x in os.listdir(os.path.join(out_folder, 'imagesTr')) if '._' not in x and '.json' not in x and 'DS_Store' not in x]
    cases = np.unique(cases)
    # -- Build fixed image, moving image and seg image pairs -- #
    pairs = [(os.path.join(out_folder, 'imagesTr', x+'_0000.nii.gz'), os.path.join(out_folder, 'imagesTr', x+'_0001.nii.gz'), os.path.join(out_folder, 'labelsTr', x+'_0001.nii.gz')) for x in cases]
    for pair in pairs:
        f, m, s = sitk.ReadImage(pair[0]), sitk.ReadImage(prototype), sitk.ReadImage(prototype_seg) # Load updated prototype here
        m = sitk.GetImageFromArray(sitk.GetArrayFromImage(m).astype(sitk.GetArrayFromImage(f).dtype))
        moving, seg = rigid_align(f, m, s)
        seg[seg != 0] = 1 # <-- Keep it binary..
        sitk.WriteImage(sitk.GetImageFromArray(moving), pair[1])    # Store at old location to overwrite the old 0001 file
        sitk.WriteImage(sitk.GetImageFromArray(seg), pair[2])
        # Update Ts folders as well!
        sitk.WriteImage(sitk.GetImageFromArray(moving), pair[1].replace('imagesTr', 'imagesTs'))    # Store at old location to overwrite the old 0001 file
        sitk.WriteImage(sitk.GetImageFromArray(seg), pair[2].replace('labelsTr', 'labelsTs'))