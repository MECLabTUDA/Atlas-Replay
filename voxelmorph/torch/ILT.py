from voxelmorph.torch.networks import Unet_
from .modelio import LoadableModel, store_config_args

class ILT(LoadableModel):
    """
    ILT network for segmentation.
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
        self.unet_model_old = Unet_(
            inshape=inshape,
            infeats=infeats,
            nb_features=nb_features,
            nb_levels=nb_levels,
            feat_mult=feat_mult,
            nb_conv_per_level=nb_conv_per_level,
            half_res=half_res,
        )
        self.unet_model_old = self.freeze_unet(self.unet_model_old)

        self.unet_model = Unet_(
            inshape=inshape,
            infeats=infeats,
            nb_features=nb_features,
            nb_levels=nb_levels,
            feat_mult=feat_mult,
            nb_conv_per_level=nb_conv_per_level,
            half_res=half_res,
        )

    def forward(self, x, kd=False):
        r"""Forward pass.
        """
        x_, x_intermediate = self.unet_model(x, True)
        if kd:
            x_o, xo_intermediate = self.unet_model_old(x, True)
            return x_, x_intermediate, x_o, xo_intermediate
        else:
            return x_

    def freeze_unet(self, unet):
        r"""Freeze U-Net.
        """
        for param in unet.parameters():
            param.requires_grad = False
        return unet
    
    def after_train(self):
        r"""This updates the old model and freezes it.
        """
        unet_new_state_dict = self.unet_model.state_dict()
        self.unet_model_old.load_state_dict(unet_new_state_dict)
        self.unet_model_old = self.freeze_unet(self.unet_model_old)