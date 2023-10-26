#!/usr/bin/env python

"""
Script to run multiple experiments in a sequence using a Telegram Bot to get update messages.
This can be used to trigger one time a list of experiments without executing them one by one.
It also checks if an experiment is finished and if so, it will skip it, else it continues with the training.
"""
import sys, os
from pythonbots.TelegramBot import TelegramBot
from scripts.torch.train_bic import train_bic as train

# -- Set configurations manually -- #
device = 0
nr_epochs = 250
mappings = {'110': 'Task110_RUNMC', '111': 'Task111_BMC', '112': 'Task112_I2CVB',
            '113': 'Task113_UCL', '114': 'Task114_BIDMC', '115': 'Task115_HK', 
            '116': 'Task116_DecathProst', 'joint': 'joint'}
train_on = [['110'],
            ['110', '111'],
            ['110', '111', '112'],
            ['110', '111', '112', '113'],
            ['110', '111', '112', '113', '114'],
            ['110', '111', '112', '113', '114', '115'],
            ['110', '111', '112', '113', '114', '115', '116']
           ]

continue_ = False
finished = False
continue_with_epoch = 0

TBot = TelegramBot(token=os.environ['tel_token'], chat_id=os.environ['tel_chat_id'])

# train.py --img-list /home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/train_list.txt 
#          --seg-list /home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/train_seg.txt
#          --model-dir /home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_110_111_112_113_ncc_ce
#          --load-model /home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_110_111_112_113_ncc_ce/0250.pt
#          --initial-epoch 10
#          --gpu 0
#          --epochs 250
#          --task 110
#          --seg

# -- Train based on the configurations -- #
TBot.send_msg("Start training based on user configurations:")
for tasks in train_on:
    trained_list = []
    for task in tasks:
        prev_mod_built = '_'.join(trained_list)
        trained_list.append(task)
        built_ts = '_'.join(trained_list)
        img_list = f'/home/aranem_locale/Desktop/MICCAI_2023/experiments/UNet_VxM_bic_7/list_files/{mappings[task]}/train_list.txt'
        seg_list = f'/home/aranem_locale/Desktop/MICCAI_2023/experiments/UNet_VxM_bic_7/list_files/{mappings[task]}/train_seg.txt'
        out_folder = f'/home/aranem_locale/Desktop/MICCAI_2023/experiments/UNet_VxM_bic_7/unet_torch_250_{built_ts}_ce'
        
        # -- Check if it is already trained or not -- #
        if os.path.exists(out_folder):
            # -- Started training on, so restore if more than one checkpoint -- #
            chks = [x for x in os.listdir(out_folder) if '.pt' in x]
            if len(chks) <= 1:  # Only 0000.pt in the list
                if len(trained_list) > 1: # <-- We still need load_model here
                    prev_model = out_folder.replace(built_ts, prev_mod_built)
                    continue_, finished, continue_with_epoch = True, True, 0
                    load_model = os.path.join(prev_model, '%04d_01.pt' % nr_epochs)    # <-- Should exist!
                else:
                    continue_, finished, continue_with_epoch = False, False, 0
            else:
                chks.sort()
                chkp = chks[-1]
                if str(nr_epochs) in chkp:
                    continue_, finished, continue_with_epoch = False, False, 0
                    TBot.send_msg(f"Model {out_folder} is already trained :)")
                    continue    # <-- Finished with training for this task
                continue_, finished, continue_with_epoch = True, False, int(chkp.split('.pt')[0][1:])
                load_model = os.path.join(out_folder, '%04d_01.pt' % continue_with_epoch)
                TBot.send_msg(f"Checkpoint %04d_01.pt found for model {out_folder} --> continue with training.." % continue_with_epoch)

        elif len(trained_list) > 1: # <-- We still need load_model here
            prev_model = out_folder.replace(built_ts, prev_mod_built)
            continue_, finished, continue_with_epoch = True, True, 0
            load_model = os.path.join(prev_model, '%04d_01.pt' % nr_epochs)    # <-- Should exist!

        # -- Build up arguments -- #
        args = [sys.argv[0], '--img-list']
        args += [img_list]
        args += ['--seg-list', seg_list]
        args += ['--model-dir', out_folder]
        args += ['--task', task]
        if continue_:
            args += ['--load-model', load_model]
            if not finished:
                args += ['--initial-epoch', str(continue_with_epoch)]
        args += ['--gpu', str(device)]
        args += ['--epochs', str(nr_epochs)]
        if len(tasks) > 1:
            args += ['--use_bias']

        # -- Train -- #
        sys.argv = args
        if continue_:
            TBot.send_msg(f"Start training {out_folder} using the segmentation network and prev_model {load_model}..")
        else:
            TBot.send_msg(f"Start training {out_folder} using the segmentation network..")
        train()
    
    TBot.send_msg(f"The training on {tasks} is succesfully completed, congrats!")
TBot.send_msg("The training is succesfully completed, congrats!")