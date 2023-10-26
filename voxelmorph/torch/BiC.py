import torch
from voxelmorph.torch.networks import Unet_
from .modelio import LoadableModel, store_config_args

class BiC(LoadableModel):
    """
    BiC U-Net network for segmentation. Uses BiC method from: https://arxiv.org/pdf/1905.13260.pdf.
    """
    @store_config_args
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """
        super().__init__()
        self.unet_model = Unet_(
            inshape=inshape,
            infeats=infeats,
            nb_features=nb_features,
            nb_levels=nb_levels,
            feat_mult=feat_mult,
            nb_conv_per_level=nb_conv_per_level,
            half_res=half_res,
        )
        # -- Add the Bias Layer -- #
        self.bias_layer = BiasLayer()

    def forward(self, x):
        r"""Forward pass.
        """
        x = self.unet_model(x)
        x = self.bias_layer(x)
        return x 

class BiasLayer(torch.nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))

    def forward(self, x):
        return self.alpha * x + self.beta
    
    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())
