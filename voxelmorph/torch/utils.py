# import os
import random
import numpy as np
import torch
import SimpleITK as sitk

def set_all_seeds(seed):
  random.seed(seed)
  # os.environ("PYTHONHASHSEED") = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def rigid_align(fixed, moving, seg_m):
    r"""All inputs should we sitk Image objects not numpy arrays.
    """
    initial_transform = sitk.CenteredTransformInitializer(fixed, moving,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY
                                                         )
    moving = sitk.Resample(moving, fixed, initial_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    # -- Use the transform between the images to apply it onto the segmentation as we don’t have GT during inference -- #
    seg_m = sitk.Resample(seg_m, fixed, initial_transform, sitk.sitkLinear, 0.0, seg_m.GetPixelID())
    return sitk.GetArrayFromImage(moving), sitk.GetArrayFromImage(seg_m)