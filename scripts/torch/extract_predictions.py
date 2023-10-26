import sys, os
import numpy as np
from tqdm import tqdm
from scripts.torch.register import register as register_single

ins = ['/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task110_RUNMC/imagesTs',
       '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task111_BMC/imagesTs',
       '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task112_I2CVB/imagesTs',
       '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task113_UCL/imagesTs',
       '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task114_BIDMC/imagesTs',
       '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task115_HK/imagesTs',
       '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task116_DecathProst/imagesTs'
       ]

models = [
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM/unet_torch_250_110_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM/unet_torch_250_111_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM/unet_torch_250_112_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM/unet_torch_250_113_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM/unet_torch_250_114_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM/unet_torch_250_115_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM/unet_torch_250_116_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM/unet_torch_250_110_111_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM/unet_torch_250_110_111_112_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM/unet_torch_250_110_111_112_113_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM/unet_torch_250_110_111_112_113_114_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM/unet_torch_250_110_111_112_113_114_115_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM/unet_torch_250_110_111_112_113_114_115_116_ce/0250.pt',
          
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_reh_7/unet_torch_250_110_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_reh_7/unet_torch_250_110_111_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_reh_7/unet_torch_250_110_111_112_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_reh_7/unet_torch_250_110_111_112_113_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_reh_7/unet_torch_250_110_111_112_113_114_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_reh_7/unet_torch_250_110_111_112_113_114_115_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_reh_7/unet_torch_250_110_111_112_113_114_115_116_ce/0250.pt',
          
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_bic_7/unet_torch_250_110_ce/0250_01.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_bic_7/unet_torch_250_110_111_ce/0250_01.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_bic_7/unet_torch_250_110_111_112_ce/0250_01.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_bic_7/unet_torch_250_110_111_112_113_ce/0250_01.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_bic_7/unet_torch_250_110_111_112_113_114_ce/0250_01.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_bic_7/unet_torch_250_110_111_112_113_114_115_ce/0250_01.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_bic_7/unet_torch_250_110_111_112_113_114_115_116_ce/0250_01.pt',
          
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_only/unet_torch_250_110_ce_ilt_kd_only/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_only/unet_torch_250_110_111_ce_ilt_kd_only/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_only/unet_torch_250_110_111_112_ce_ilt_kd_only/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_only/unet_torch_250_110_111_112_113_ce_ilt_kd_only/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_only/unet_torch_250_110_111_112_113_114_ce_ilt_kd_only/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_only/unet_torch_250_110_111_112_113_114_115_ce_ilt_kd_only/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_only/unet_torch_250_110_111_112_113_114_115_116_ce_ilt_kd_only/0250.pt',
          
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/VoxelMorph/VoxelMorph_rigid/vxm_torch_250_110_ncc_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/VoxelMorph/VoxelMorph_rigid/vxm_torch_250_111_ncc_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/VoxelMorph/VoxelMorph_rigid/vxm_torch_250_112_ncc_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/VoxelMorph/VoxelMorph_rigid/vxm_torch_250_113_ncc_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/VoxelMorph/VoxelMorph_rigid/vxm_torch_250_114_ncc_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/VoxelMorph/VoxelMorph_rigid/vxm_torch_250_115_ncc_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/VoxelMorph/VoxelMorph_rigid/vxm_torch_250_116_ncc_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/VoxelMorph/VoxelMorph_rigid/vxm_torch_250_110_111_ncc_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/VoxelMorph/VoxelMorph_rigid/vxm_torch_250_110_111_112_ncc_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/VoxelMorph/VoxelMorph_rigid/vxm_torch_250_110_111_112_113_ncc_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/VoxelMorph/VoxelMorph_rigid/vxm_torch_250_110_111_112_113_114_ncc_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/VoxelMorph/VoxelMorph_rigid/vxm_torch_250_110_111_112_113_114_115_ncc_ce/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/VoxelMorph/VoxelMorph_rigid/vxm_torch_250_110_111_112_113_114_115_116_ncc_ce/0250.pt',
          
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/EWC/UNet_VxM_ewc_2-2/unet_torch_250_110_ce_ewc/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/EWC/UNet_VxM_ewc_2-2/unet_torch_250_110_111_ce_ewc/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/EWC/UNet_VxM_ewc_2-2/unet_torch_250_110_111_112_ce_ewc/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/EWC/UNet_VxM_ewc_2-2/unet_torch_250_110_111_112_113_ce_ewc/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/EWC/UNet_VxM_ewc_2-2/unet_torch_250_110_111_112_113_114_ce_ewc/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/EWC/UNet_VxM_ewc_2-2/unet_torch_250_110_111_112_113_114_115_ce_ewc/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/EWC/UNet_VxM_ewc_2-2/unet_torch_250_110_111_112_113_114_115_116_ce_ewc/0250.pt',
          
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/RWalk/UNet_VxM_rwalk_1-7_0-9/unet_torch_250_110_ce_rwalk/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/RWalk/UNet_VxM_rwalk_1-7_0-9/unet_torch_250_110_111_ce_rwalk/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/RWalk/UNet_VxM_rwalk_1-7_0-9/unet_torch_250_110_111_112_ce_rwalk/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/RWalk/UNet_VxM_rwalk_1-7_0-9/unet_torch_250_110_111_112_113_ce_rwalk/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/RWalk/UNet_VxM_rwalk_1-7_0-9/unet_torch_250_110_111_112_113_114_ce_rwalk/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/RWalk/UNet_VxM_rwalk_1-7_0-9/unet_torch_250_110_111_112_113_114_115_ce_rwalk/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/RWalk/UNet_VxM_rwalk_1-7_0-9/unet_torch_250_110_111_112_113_114_115_116_ce_rwalk/0250.pt',

        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_mse_only/unet_torch_250_110_ce_ilt_mse_only/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_mse_only/unet_torch_250_110_111_ce_ilt_mse_only/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_mse_only/unet_torch_250_110_111_112_ce_ilt_mse_only/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_mse_only/unet_torch_250_110_111_112_113_ce_ilt_mse_only/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_mse_only/unet_torch_250_110_111_112_113_114_ce_ilt_mse_only/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_mse_only/unet_torch_250_110_111_112_113_114_115_ce_ilt_mse_only/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_mse_only/unet_torch_250_110_111_112_113_114_115_116_ce_ilt_mse_only/0250.pt',

        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_mse/unet_torch_250_110_ce_ilt_kd_mse/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_mse/unet_torch_250_110_111_ce_ilt_kd_mse/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_mse/unet_torch_250_110_111_112_ce_ilt_kd_mse/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_mse/unet_torch_250_110_111_112_113_ce_ilt_kd_mse/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_mse/unet_torch_250_110_111_112_113_114_ce_ilt_kd_mse/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_mse/unet_torch_250_110_111_112_113_114_115_ce_ilt_kd_mse/0250.pt',
        #   '/media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_predictions/UNet/UNet_VxM_ilt_kd_mse/unet_torch_250_110_111_112_113_114_115_116_ce_ilt_kd_mse/0250.pt',
          ]

# -- Example inputs for reference -- #
# --moving /media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_raw_data/nnUNet_ready_with_ts_folder/Task110_RUNMC/imagesTr/Case08_0001.nii.gz
# --fixed /media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_raw_data/nnUNet_ready_with_ts_folder/Task110_RUNMC/imagesTr/Case08_0000.nii.gz
# --moving_seg /media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_raw_data/nnUNet_ready_with_ts_folder_BACKUP/Task110_RUNMC/labelsTr/Case08_0001.nii.gz
# --model home/aranem_locale/Desktop/WACV_2024/WACV_2024_predictions/UNet_VxM_bic_7/vxm_torch_250_110_ncc_dice/0200_01.pt
# --out home/aranem_locale/Desktop/WACV_2024/WACV_2024_predictions/UNet_VxM_bic_7/vxm_torch_250_110_ncc_dice/eval/Case08/
# --gt /media/aranem_locale/AR_subs_exps/WACV_2024/WACV_2024_raw_data/nnUNet_ready_with_ts_folder_BACKUP/Task110_RUNMC/labelsTr/Case08_0000.nii.gz

# Extract predictions
for model in models:
    bic, seg, ilt = False, False, False
    if 'unet' in model:
        seg = True
    else:
        seg = False
    if 'bic' in model:
        bic = True
        seg = True
    if 'ilt' in model:
        ilt = True
        seg = True
    for inp in ins:
        print(f"Creating predictions with {model.split(os.sep)[-2]} for {inp.split(os.sep)[-2]}:")
        out_ = os.path.join(os.path.sep, *model.split(os.path.sep)[:-1], 'predictions', inp.split(os.sep)[-2])
        cases = [x[:-12] for x in os.listdir(inp) if '._' not in x and '.json' not in x and 'DS_Store' not in x]
        cases = np.unique(cases)
        for case in tqdm(cases):
            fixed = os.path.join(inp, case+'_0000.nii.gz')
            moving = os.path.join(inp, case+'_0001.nii.gz')
            moving_seg = os.path.join(inp.replace('imagesTs', 'labelsTs'), case+'_0001.nii.gz')
            gt = os.path.join(inp.replace('imagesTs', 'labelsTs'), case+'_0000.nii.gz')
            out = os.path.join(out_, case)
            os.makedirs(out, exist_ok=True)

            # -- Build up arguments and do registration -- #
            args = [sys.argv[0], '--model']
            args += [model, '--fixed']
            args += [fixed, '--moving']
            args += [moving, '--moving_seg']
            args += [moving_seg, '--gt']
            args += [gt, '--out']
            args += [out]
            if seg:
                args += ['--seg']
            if bic:
                args += ['--bic']
            if ilt:
                args += ['--ilt']
            sys.argv = args
            register_single()