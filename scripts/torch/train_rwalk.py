#!/usr/bin/env python

"""
Example script to train a VoxelMorph model using EWC method.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

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
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


"""
Code has been adapted so segmentations are warped during training as well. Build the list files manually and split them as well so we can make proper
validation at the end. Don't forget to set model-dir correctly.
Remember that the input images has to be affinely aligned. The moving seg is anligned using te image transformation flow, i.e. without considering the GT.
Normalization between [0, 1] takes place during training!
"""

EPSILON = 1e-8
ALPHA = 0.9
LAMBDA = 0.4

import os, copy, pickle
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm 

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

def mean_dice_coef(y_true, y_pred_bin, num_classes=1, do_torch=False):
    # from: https://www.codegrepper.com/code-examples/python/dice+similarity+coefficient+python
    # shape of y_true and y_pred_bin: (n_samples, height, width, n_channels)
    batch_size = y_true.shape[0]
    depth = y_true.shape[-1]
    # channel_num = y_true.shape[-1]
    mean_dice_channel = 0.
    # dict contains label: dice per batch
    channel_dices_per_batch = {i+1:list() for i in range(num_classes)}
    for i in range(batch_size):
        # for n in range(depth):
        for j in range(1, num_classes+1):
            y_t = y_true[i, ...].clone() if do_torch else copy.deepcopy(y_true[i, ...])
            y_p = y_pred_bin[i, ...].clone() if do_torch else copy.deepcopy(y_pred_bin[i, ...])
            y_t[y_t != j] = 0
            y_t[y_t == j] = 1
            y_p[y_p != j] = 0
            y_p[y_p == j] = 1
            channel_dice = single_dice_coef(y_t, y_p, do_torch)
            channel_dices_per_batch[j].append(channel_dice)
            # channel_dice = single_dice_coef(y_true[i, :, :, j], y_pred_bin[i, :, :, j], num_classes, do_torch)
            mean_dice_channel += channel_dice/(num_classes*batch_size)
    return mean_dice_channel, channel_dices_per_batch

def single_dice_coef(y_true, y_pred_bin, do_torch=False):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin) if not do_torch else torch.sum(y_true * y_pred_bin)
    if do_torch:
        if (torch.sum(y_true)==0) and (torch.sum(y_pred_bin)==0):
            return 1
        return ((2*intersection) / (torch.sum(y_true) + torch.sum(y_pred_bin))).item()
    else:
        if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
            return 1
        return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

def train_rwalk():
    # parse the commandline
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
    parser.add_argument('--task', required=True, help='Task Name we train on')
    parser.add_argument('--img-prefix', help='optional input image file prefix')
    parser.add_argument('--img-suffix', help='optional input image file suffix')
    parser.add_argument('--seg-list', required=True, help='line-seperated list of training segs')
    parser.add_argument('--seg-prefix', help='optional input seg file prefix')
    parser.add_argument('--seg-suffix', help='optional input seg file suffix')
    # parser.add_argument('--model-dir', default='/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_113_ncc_ce',
    parser.add_argument('--model-dir', default='/home/aranem_locale/Desktop/MICCAI_2023/experiments/VoxelMorph_rigid/vxm_torch_250_110_111_112_113_ncc_ce',
    # parser.add_argument('--model-dir', default='/home/aranem_locale/Desktop/MICCAI_2023/experiments/UNet_VxM/unet_torch_250_110_ce',
                        help='model output directory.')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')

    # training parameters
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=250,
                        help='number of training epochs (default: 250)')
    parser.add_argument('--steps-per-epoch', type=int, default=250,
                        help='frequency of model saves (default: 100)')
    parser.add_argument('--load-model', help='optional model file to initialize with')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--cudnn-nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')
    parser.add_argument('--enc', type=int, nargs='+',
                        help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7,
                        help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')

    args = parser.parse_args()

    # load and prepare training data
    train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                            suffix=args.img_suffix)
    seg_files = vxm.py.utils.read_file_list(args.seg_list, prefix=args.seg_prefix,
                                            suffix=args.seg_suffix)
    assert len(train_files) > 0, 'Could not find any training data.'
    assert len(seg_files) > 0, 'Could not find any segmentations.'

    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = not args.multichannel

    # scan-to-scan generator
    generator = vxm.generators.scan_to_scan(
        train_files, seg_files, batch_size=args.batch_size, bidir=False, add_feat_axis=add_feat_axis)

    # extract shape from sampled input
    inshape = next(generator)[0][0].shape[1:-1]

    # prepare model folder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert np.mod(args.batch_size, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_devices)

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    # unet architecture
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    if args.load_model:
        # load initial model (if specified)
        model = vxm.networks.UNet.load(args.load_model, device)
        fisher = load_pickle(os.path.join(os.sep, *args.load_model.split(os.sep)[:-1], 'fisher.pkl'))
        params = load_pickle(os.path.join(os.sep, *args.load_model.split(os.sep)[:-1], 'params.pkl'))
        scores = load_pickle(os.path.join(os.sep, *args.load_model.split(os.sep)[:-1], 'scores.pkl'))
        # prev_param = {k: torch.clone(v).detach().cpu() for k, v in model.named_parameters() if v.grad is not None}
    else:
        fisher, params, scores = dict(), dict(), dict()
        model = vxm.networks.UNet(
            inshape=inshape,
            infeats=1,  # <-- One input feature
            nb_features=[enc_nf, dec_nf],
            nb_levels=None,
            feat_mult=1,
            feat_out=2, # <-- 2 output features (binary)
            nb_conv_per_level=1,
            half_res=False,
        )
    prev_param = None

    # -- Define the fisher and params before the training -- #
    fisher[args.task] = {n: torch.zeros_like(p, device='cuda:0', requires_grad=False) for n, p in model.named_parameters() if p.requires_grad}
    params[args.task] = dict()
    scores[args.task] = {n: torch.zeros_like(p, device='cuda:0', requires_grad=False) for n, p in model.named_parameters() if p.requires_grad}
    
    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # prepare image loss
    losses = [torch.nn.CrossEntropyLoss()]
    weights = [1]

    if len([x for x in fisher.keys() if x != args.task]) > 0:
        losses.append(vxm.losses.RWalkLoss(fisher=fisher, params=params, parameter_importance=scores, ewc_lambda=LAMBDA))
        weights.append(1)

    # training loops
    for epoch in range(args.initial_epoch, args.epochs):

        # save model checkpoint
        if epoch % 20 == 0:
            model.save(os.path.join(model_dir, '%04d.pt' % epoch))
            
            # -- Update the importance score using distance in Riemannian Manifold -- #
            if prev_param is not None:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # -- Get parameter difference from old param and current param t -- #
                        delta = param.grad.detach() * (prev_param[name].to(param.device) - param.detach())
                        delta = delta.to(0)
                        # -- Calculate score denominator -- #
                        den = 0.5 * fisher[args.task][name] * (param.detach() - prev_param[name].to(param.device)).pow(2).to(0) + EPSILON
                        # -- Score: delat(L) / 0.5*F_t*delta(param)^2 --> only positive or zero values -- #
                        scores_ = (delta / den)
                        scores_[scores_ < 0] = 0  # Ensure no negative values
                        # -- Update the scores -- #
                        scores[args.task][name] += scores_

            # -- Update the prev params -- #
            if epoch != 0:
                prev_param = {k: torch.clone(v).detach().cpu() for k, v in model.named_parameters() if v.grad is not None}

            # -- Update the fisher values -- #
            for name, param in model.named_parameters():
                # -- F_t = alpha * F_t + (1-alpha) * F_t-1
                if param.grad is not None:
                    f_t = param.grad.data.clone().pow(2).to(0)
                    f_to = fisher[args.task][name] if fisher[args.task][name] is not None else torch.tensor([0], device='cuda:0')
                    fisher[args.task][name] = (ALPHA * f_t) + ((1 - ALPHA) * f_to)

            for name, param in model.named_parameters():
                # -- Update the params dict -- #
                params[args.task][name] = param.data.clone()

            write_pickle(fisher, os.path.join(model_dir, 'fisher.pkl'))
            write_pickle(params, os.path.join(model_dir, 'params.pkl'))
            write_pickle(scores, os.path.join(model_dir, 'scores.pkl'))

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []
        dice = []

        # for step in range(steps):
        for _ in range(args.steps_per_epoch):
        # for step in tqdm(range(args.steps_per_epoch)):

            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true, segs = next(generator)
            inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
            segs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in segs]

            loss = 0
            loss_list = []

            # run inputs through the model to produce a warped image and flow field
            y_pred_ = model(inputs[1])   # <-- Only target into U-net, segmentation out
            # Switch first entries so CE loss gets pred first and then the target, otherwise it gets the wrong arguments!
            y_pred = [segs[1].squeeze(1).long(), torch.nn.functional.softmax(y_pred_, dim=1).float()]
            y_true = [y_pred_]

            # calculate total loss
            for n, loss_function in enumerate(losses):
                if n == 1: # RWalk Loss
                    curr_loss = loss_function.loss(model.named_parameters()) * weights[n]
                else:
                    curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())
                
            # -- Calculate the Dices between moved image and fixed image as well as not moved image and fixed image -- #
            _, channel_dices_per_batch = mean_dice_coef(segs[1], y_pred[-1].argmax(1), 1, True)
            
            mean_dice = [np.mean(v) for _, v in channel_dices_per_batch.items()] # Dice between moved and fixed segmentation
            dice.append(mean_dice)

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

            # Create first prev_param after first step as its None otherwise
            if prev_param is None and epoch == 0:
                prev_param = {k: torch.clone(v).detach().cpu() for k, v in model.named_parameters() if v.grad is not None}

        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        dice_info = 'dice: %.4f' % (np.mean(dice))
        print(' - '.join((epoch_info, time_info, loss_info, dice_info)), flush=True)

    # final model save
    model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))

    # -- Extract fisher und param values -- #
    model.train()
    optimizer.zero_grad()
    for _ in range(args.steps_per_epoch):
        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true, segs = next(generator)
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
        segs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in segs]

        loss = 0
        loss_list = []

        # run inputs through the model to produce a warped image and flow field
        y_pred_ = model(inputs[1])   # <-- Only target into U-net, segmentation out
        # Switch first entries so CE loss gets pred first and then the target, otherwise it gets the wrong arguments!
        y_pred = [segs[1].squeeze(1).long(), torch.nn.functional.softmax(y_pred_, dim=1).float()]
        y_true = [y_pred_]

        # calculate total loss
        for n, loss_function in enumerate(losses):
            if n == 1: # EWC Loss
                curr_loss = loss_function.loss(model.named_parameters()) * weights[n]
            else:
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate but NO optimization!
        optimizer.zero_grad()
        loss.backward()
    

    # -- Update the importance score one last time once finished training using distance in Riemannian Manifold -- #
    for name, param in model.named_parameters():
        if param.grad is not None:
            # -- Get parameter difference from old param and current param t -- #
            delta = param.grad.detach() * (prev_param[name].to(param.device) - param.detach())
            delta = delta.to(0)
            # -- Calculate score denominator -- #
            den = 0.5 * fisher[args.task][name] * (param.detach() - prev_param[name].to(param.device)).pow(2).to(0) + EPSILON
            # -- Score: delat(L) / 0.5*F_t*delta(param)^2 --> only positive or zero values -- #
            scores_ = (delta / den)
            scores_[scores_ < 0] = 0  # Ensure no negative values
            # -- Update the scores -- #
            scores[args.task][name] += scores_

    # -- Store params -- #
    for name, param in model.named_parameters():
        # -- Update the params dict -- #
        params[args.task][name] = param.data.clone()

    # -- Update the fisher values -- #
    for name, param in model.named_parameters():
        # -- F_t = alpha * F_t + (1-alpha) * F_t-1
        if param.grad is not None:
            f_t = param.grad.data.clone().pow(2).to(0)
            f_to = fisher[args.task][name] if fisher[args.task][name] is not None else torch.tensor([0], device='cuda:0')
            fisher[args.task][name] = (ALPHA * f_t) + ((1 - ALPHA) * f_to)

    # -- Normalize the fisher values to be in range 0 to 1 -- #
    values = [torch.max(val) for val in scores[args.task].values()] # --> only for the current task of course
    minim, maxim = min(values), max(values)
    for k, v in fisher[args.task].items():
        fisher[args.task][k] = (v - minim) / (maxim - minim + EPSILON)

    # -- Normalize the score values to be in range 0 to 1 -- #
    values = [torch.max(val) for val in scores[args.task].values()] # --> only for the current task of course
    minim, maxim = min(values), max(values)
    
    if len([x for x in scores.keys() if x != args.task]) > 0:
        # -- Average current and previous scores -- #
        prev_scores = {k: v.clone() for k, v in scores[list(scores.keys())[-1]].items()}
        for k, v in scores[args.task].items():
            # -- Normalize the score -- #
            curr_score_norm = (v - minim) / (maxim - minim + EPSILON)
            # -- Average the score to alleviate rigidity due to the accumulating sum of the scores otherwise -- #
            scores[args.task][k] = 0.5 * (prev_scores[k] + curr_score_norm)
    else:
        # -- Only average current scores -- #
        for k, v in scores[args.task].items():
            # -- Normalize and scale the score so that division does not have an effect -- #
            curr_score_norm = (v - minim) / (maxim - minim + EPSILON)
            scores[args.task][k] = 2 * curr_score_norm

    write_pickle(fisher, os.path.join(model_dir, 'fisher.pkl'))
    write_pickle(params, os.path.join(model_dir, 'params.pkl'))
    write_pickle(scores, os.path.join(model_dir, 'scores.pkl'))

# -- Main function for setup execution -- #
def main():
    train_rwalk()

if __name__ == "__main__":
    train_rwalk()