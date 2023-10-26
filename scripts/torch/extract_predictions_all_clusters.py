import sys, os
import numpy as np
from tqdm import tqdm
from scripts.torch.register import register as register_single

ins = ['/home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task110_RUNMC/imagesTs',
       '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task111_BMC/imagesTs',
       '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task112_I2CVB/imagesTs',
       '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task113_UCL/imagesTs',
       '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task114_BIDMC/imagesTs',
       '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task115_HK/imagesTs',
       '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/clusters_rigid/nnUNet_ready_with_ts_folder_rigid_aligned/Task116_DecathProst/imagesTs']

models = [
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_110_ncc_ce/0250.pt',
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_111_ncc_ce/0250.pt',
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_112_ncc_ce/0250.pt',
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_113_ncc_ce/0250.pt',
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_114_ncc_ce/0250.pt',
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_115_ncc_ce/0250.pt',
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_116_ncc_ce/0250.pt',
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_joint_ncc_ce/0250.pt',
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_110_111_ncc_ce/0250.pt',
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_110_111_112_ncc_ce/0250.pt',
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_110_111_112_113_ncc_ce/0250.pt',
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_110_111_112_113_114_ncc_ce/0250.pt',
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_110_111_112_113_114_115_ncc_ce/0250.pt',
          '/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_110_111_112_113_114_115_116_ncc_ce/0250.pt'
          ]

mapping = {'/home/aranem_locale/Desktop/MICCAI_2023/experiments/Clusters_representations_rigid/no_coil/Case_002.nii.gz': 'no_coil',
           '/home/aranem_locale/Desktop/MICCAI_2023/experiments/Clusters_representations_rigid/coil_1_bright/Case_001.nii.gz': 'coil_1_bright',
           '/home/aranem_locale/Desktop/MICCAI_2023/experiments/Clusters_representations_rigid/coil_2_mid/Case_000.nii.gz': 'coil_2_mid',
           '/home/aranem_locale/Desktop/MICCAI_2023/experiments/Clusters_representations_rigid/coil_3_dark/Case_001.nii.gz': 'coil_3_dark'}

clusters = ['/home/aranem_locale/Desktop/MICCAI_2023/experiments/Clusters_representations_rigid/no_coil/Case_002.nii.gz',
            '/home/aranem_locale/Desktop/MICCAI_2023/experiments/Clusters_representations_rigid/coil_1_bright/Case_001.nii.gz',
            '/home/aranem_locale/Desktop/MICCAI_2023/experiments/Clusters_representations_rigid/coil_2_mid/Case_000.nii.gz',
            '/home/aranem_locale/Desktop/MICCAI_2023/experiments/Clusters_representations_rigid/coil_3_dark/Case_001.nii.gz']

# -- Example inputs for reference -- #
# --moving /home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder/Task110_RUNMC/imagesTr/Case08_0001.nii.gz
# --fixed /home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder/Task110_RUNMC/imagesTr/Case08_0000.nii.gz
# --moving_seg /home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder_BACKUP/Task110_RUNMC/labelsTr/Case08_0001.nii.gz
# --model home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_110_ncc_ce/0200.pt
# --out home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_110_ncc_ce/eval/Case08/
# --gt /home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/MICCAI_2023/MICCAI_2023_raw_data/nnUNet_ready_with_ts_folder_BACKUP/Task110_RUNMC/labelsTr/Case08_0000.nii.gz

# Extract predictions
for model in models:
    for inp in ins:
        # Loop through all clusters to get a comparison against all clusters
        for moving in clusters:
            print(f"Creating predictions with {model.split(os.sep)[-2]} for {inp.split(os.sep)[-2]} using cluster {mapping[moving]}:")
            out_ = os.path.join(os.path.sep, *model.split(os.path.sep)[:-1], 'predictions', mapping[moving], inp.split(os.sep)[-2])
            cases = [x[:-12] for x in os.listdir(inp) if '._' not in x and '.json' not in x and 'DS_Store' not in x]
            cases = np.unique(cases)
            for case in tqdm(cases):
                fixed = os.path.join(inp, case+'_0000.nii.gz')
                moving_seg = os.path.join(moving.replace('.nii.gz', '_seg.nii.gz'))
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
                args += ['--do_rigid']
                sys.argv = args
                register_single()