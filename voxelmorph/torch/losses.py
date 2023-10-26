import numpy as np
import torch, copy, math
import torch.nn.functional as F


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

# -- From: https://github.com/MECLabTUDA/Lifelong-nnUNet/blob/f22c01eebc2b0c542ca0a40ac722f90aab05fc54/nnunet_ext/training/loss_functions/deep_supervision.py#L86 -- #
class RWalkLoss:
    """
    RWalk Loss.
    """
    def __init__(self, ewc_lambda=0.4, fisher=dict(), params=dict(), parameter_importance=dict()):
        self.ewc_lambda = ewc_lambda
        self.tasks = list(fisher.keys())[:-1]   # <-- Current task is already in there not like simple EWC!
        self.fisher = fisher
        self.params = params
        self.parameter_importance = parameter_importance

    def loss(self, network_params):
        # -- Update the network_params -- #
        loss = 0
        for task in self.tasks:
            for name, param in network_params: # Get named parameters of the current model
                # -- Extract corresponding fisher and param values -- #
                param_ = param
                fisher_value = self.fisher[task][name]
                param_value = self.params[task][name]
                importance = self.parameter_importance[task][name]
                
                # -- loss = loss_{t} + ewc_lambda * \sum_{i} (F_{i} + S(param_{i})) * (param_{i} - param_{t-1, i})**2 -- #
                loss += self.ewc_lambda * ((fisher_value + importance) * (param_ - param_value).pow(2)).sum()
        return loss

# -- From: https://github.com/MECLabTUDA/Lifelong-nnUNet/blob/f22c01eebc2b0c542ca0a40ac722f90aab05fc54/nnunet_ext/training/loss_functions/deep_supervision.py#L15 -- #
class EWCLoss:
    """
    EWC Loss.
    """
    def __init__(self, ewc_lambda=0.4, fisher=dict(), params=dict()):
        self.ewc_lambda = ewc_lambda
        self.tasks = list(fisher.keys())
        self.fisher = fisher
        self.params = params

    def loss(self, network_params):
        # -- Update the network_params -- #
        loss = 0
        for task in self.tasks:
            for name, param in network_params: # Get named parameters of the current model
                # -- Extract corresponding fisher and param values -- #
                # fisher_value = self.fisher[task][name]
                # param_value = self.params[task][name]
                param_ = param
                fisher_value = self.fisher[task][name]
                param_value = self.params[task][name]
                # loss = to_cuda(loss, gpu_id=param.get_device())
                
                # -- loss = loss_{t} + ewc_lambda/2 * \sum_{i} F_{i}(param_{i} - param_{t-1, i})**2 -- #
                loss += self.ewc_lambda/2 * (fisher_value * (param_ - param_value).pow(2)).sum()
        return loss
    
class ILTLoss:
    """
    ILT Loss.
    """
    def __init__(self, use_feats=False):
        self.use_feats = use_feats

    def loss(self, y_pred_, y_true, y_pred_old, y_pred_interm, y_pred_old_interm):
        r"""Don't forget to compute the CE Loss as well.
        """
        # Assuming new_softmax and old_softmax are tensors of shape (batch_size, num_classes, height, width)
        loss_kd = F.cross_entropy(F.softmax(y_pred_, dim=1).float(), F.softmax(y_pred_old, dim=1).float())
        # Don't need to mask as we use all classes, we always only use one class
        # target_mask = (y_true <= (1))#.squeeze(1)   <-- Not necessary as we always use all classes here, don't need to mask
        # loss_masked = loss_kd[target_mask]
        # loss_kd = loss_masked.mean()

        # Replace NaN values with 0
        loss_kd = torch.where(torch.isnan(loss_kd), torch.tensor(0.).to(loss_kd.device), loss_kd)

        # -- Second approach uses intermediate results for MSE Loss -- #
        loss_mse = F.mse_loss(y_pred_interm, y_pred_old_interm)

        if self.use_feats == 'both':
            return loss_mse + loss_kd
        elif self.use_feats:
            return loss_mse
        else:
            return loss_kd



# -- From: https://github.com/MECLabTUDA/Lifelong-nnUNet/blob/f22c01eebc2b0c542ca0a40ac722f90aab05fc54/nnunet_ext/training/loss_functions/deep_supervision.py#L217 -- #
class PLOPLoss:
    """
    PLOP Loss.
    """
    def __init__(self, nr_classes=1, pod_lambda=0.01, scales=3):
        # -- This implementation represents the method proposed in the paper https://arxiv.org/pdf/2011.11390.pdf -- #
        # -- Based on the implementation from here: https://github.com/arthurdouillard/CVPR2021_PLOP/blob/main/train.py -- #
        # -- Set all variables that are used by parent class and are necessary for the EWC loss calculation -- #
        self.scales = scales
        self.nr_classes = nr_classes
        self.pod_lambda = pod_lambda
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    def update_plop_params(self, old_interm_results, interm_results, thresholds, max_entropy):
        r"""The old_interm_results and interm_results should be updated before calculating the loss (every batch).
        """
        # -- Update the convolutional outputs -- #
        self.thresholds = thresholds        # Structure: {seg_outputs_ID: thresholds_tensor, ...}
        self.max_entropy = max_entropy
        self.interm_results = interm_results
        self.old_interm_results = old_interm_results    # Structure: {layer_name: embedding, ...}
        self.num_layers = len(self.old_interm_results.keys())
    
    def _pseudo_label_loss(self, x, x_o, y, idx=1):
        r"""This function calculates the pseudo label loss using entropy dealing with the background shift.
            x_o should be the old models prediction and idx the index of the current selected output --> do not forget to detach x_o!
        """
        # -- Define the background mask --> everything that is 0 -- #
        labels = copy.deepcopy(y)
        mask_background = labels == 0

        # -- Calculate the softmax of the old output -- #
        probs = torch.softmax(x_o, dim=1)
        # -- Extract the pseudo labels -- #
        _, pseudo_labels = probs.max(dim=1)
        # -- Extract the valid pseudo mask -- #
        mask_valid_pseudo = (entropy(probs) / self.max_entropy) < self.thresholds[idx][pseudo_labels]

        # mask_valid_pseudo = (entropy(probs) / self.max_entropy) < self.thresholds[idx][pseudo_labels]
        # -- Don't consider all labels that are not confident enough -- #
        labels[~mask_valid_pseudo & mask_background] = 255
        # -- Consider all other labels as pseudo ones -- #
        labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo & mask_background].float()
        # -- Extract the number of certain background pixels -- #
        num = (mask_valid_pseudo & mask_background).float().sum(dim=(1,2))
        # -- Total number of bacground pixels -- #
        den =  mask_background.float().sum(dim=(1,2))
        # -- Calculate the adaptive factor -- #
        classif_adaptive_factor = num / den
        classif_adaptive_factor = classif_adaptive_factor[:, None, None]
        
        # -- NOT Pseudo Loss -- #
        # -- Calculate the unbiased CE for the non pseudo labels --> Now use the actual output of current model -- #
        mask = mask_background & mask_valid_pseudo
        lab = copy.deepcopy(y)
        if mask is not None:
            lab[mask] = 255
        loss_not_pseudo = self.ce(x, lab.long())

        # -- Pseudo Loss -- #
        # -- Prepare labels for the pseudo loss -- #
        _labels = copy.deepcopy(y)
        _labels[~(mask_background & mask_valid_pseudo)] = 255
        _labels[mask_background & mask_valid_pseudo] = pseudo_labels[mask_background & mask_valid_pseudo].float()
        # -- Calculate the pseudo loss -- #
        loss_pseudo = self.ce(x, _labels)
        
        # -- Return the joined loss -- #
        loss = classif_adaptive_factor * (loss_pseudo + loss_not_pseudo)
        return loss.mean()
    
    def pod_embed(self, embedding_tensor):
        # -- Calculate the POD embedding -- #
        w_p = torch.mean(embedding_tensor, -1)  # Over W: H × C width-pooled slices of embedding_tensor using mean
        h_p = torch.mean(embedding_tensor, -2)  # Over H: W × C height-pooled slices of embedding_tensor using mean
        return torch.cat((w_p, h_p), dim=1)     # Concat over C axis

    def local_POD(self, h_, h_old, scales):
        # -- Calculate the local POD embedding using intermediate convolutional outputs -- #
        assert h_.size() == h_old.size(), "The embedding tensors of the current and old model should have the same shape.."
        
        # -- Initialize the embedding lists/tensors that are filled in the double for loop -- #
        POD_ = None
        POD_old = None
        # -- Extract the height and width of the current embeddings -- #
        W = h_.size(-1)
        H = h_.size(-2)
        # -- Calculate embeddings for every scale in scales -- #
        for scale in range(0, scales, 1):  # step size = 1
            # -- Calculate step sizes -- #
            w = int(W/(2**scale))
            h = int(H/(2**scale))

            # -- Throw an error if scale is too big resulting in a step size of 0 -- #
            assert w > 0 and h > 0,\
                "The number of scales ({}) are too big in such a way that during scale {} either the step size for H ({}) or W ({}) is 0..".format(scales, scale, h, w)

            # -- Loop through W and H in h and w steps -- #
            for i in range(0, W-w, w):
                for j in range(0, H-h, h):
                    # -- Calculate the POD embeddings for the extracted slice based on i and j -- #
                    pod_ = self.pod_embed(h_[..., i:i+w, j:j+h])
                    pod_old = self.pod_embed(h_old[..., i:i+w, j:j+h])
                    # -- In-Place concatenation of the POD embeddings along channels axis --> use last one sine they are different -- #
                    POD_ = pod_ if POD_ is None else torch.cat((POD_, pod_), dim=-1)                  # concat over last dim since those might be different
                    POD_old = pod_old if POD_old is None else torch.cat((POD_old, pod_old), dim=-1)   # concat over last dim since those might be different

        # -- Return the L2 distance between the POD embeddings based on their original implementation from here: -- #
        # -- https://github.com/arthurdouillard/CVPR2021_PLOP/blob/0fb13774735961a6cb50ccfee6ca99d0d30b27bc/train.py#L934 -- #
        layer_loss = torch.stack([torch.linalg.norm(p_ - p_o, dim=-1) for p_, p_o in zip(POD_, POD_old)])
        return torch.mean(layer_loss)
    
    def loss(self, x, x_o, y):
        pseudo_loss = self._pseudo_label_loss(x, x_o, y, idx=1)   # <-- We only have on class so there is only one threshold

        # pseudo_loss = self._pseudo_label_loss(x[0], x_o[0], y[0].squeeze(), idx=0)
        # for i in range(1, len(x)):
        #     if weights[i] != 0:
        #         pseudo_loss += weights[i] * self._pseudo_label_loss(x[i], x_o[i], y[i].squeeze(), idx=i)
                
        # -- Update the loss as proposed in the paper and return this loss to the calling function instead -- #
        dist_loss = 0
        for name, h_old in self.old_interm_results.items(): # --> Loop over every Layer
            # h_old = to_cuda(h_old, gpu_id=self.interm_results[name].device)
            # -- Add the local POD loss as distillation loss ontop of the original loss value -- #
            dist_loss += self.pod_lambda * self.local_POD(self.interm_results[name], h_old, self.scales)
            # -- NOTE: The adaptive weighting \sqrt(|C^{1:t}| / |C^{t}|) is not necessary for us -- #
            # --       since we always have the same classes resulting in a weighting of 1 -- #
            
            # -- Update the loss a final time -- #
            dist_loss /= self.num_layers # Divide by the number of layers we looped through
        
        # -- Empty the variable that hold the data -- #
        del self.thresholds, self.max_entropy, self.interm_results, self.old_interm_results

        # -- Return the updated loss value -- #
        # pseudo_loss = to_cuda(pseudo_loss, gpu_id=dist_loss.device)
        return pseudo_loss + dist_loss

def entropy(probabilities):
    r"""Computes the entropy per pixel.
    # References:
        * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
        Saporta et al.
        CVPR Workshop 2020
    :param probabilities: Tensor of shape (b, c, w, h).
    :return: One entropy per pixel, shape (b, w, h)
    """
    factor = 1 / math.log(probabilities.shape[1] + 1e-8)
    return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)
    
# -- From: https://github.com/camgbus/medical_pytorch/blob/ef1b029e78a6fc5db07d127df375fef3b2c5b7a2/mp/eval/losses/loss_abstract.py#L9 -- #
class LossAbstract(torch.nn.Module):
    r"""A named loss function, that loss functions should inherit from.
        Args:
            device (str): device key
    """
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
        self.name = self.__class__.__name__

    def get_evaluation_dict(self, output, target):
        r"""Return keys and values of all components making up this loss.
        Args:
            output (torch.tensor): a torch tensor for a multi-channeled model 
                output
            target (torch.tensor): a torch tensor for a multi-channeled target
        """
        return {self.name: float(self.forward(output, target).cpu())}
        
class LossDice(LossAbstract):
    r"""Dice loss with a smoothing factor."""
    def __init__(self, smooth=1., device='cuda:0'):
        super().__init__(device=device)
        self.smooth = smooth
        self.name = 'LossDice[smooth='+str(self.smooth)+']'

    def forward(self, target, output):
        output_flat = output.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (output_flat * target_flat).sum()
        return 1 - ((2. * intersection + self.smooth) /
                (output_flat.sum() + target_flat.sum() + self.smooth))