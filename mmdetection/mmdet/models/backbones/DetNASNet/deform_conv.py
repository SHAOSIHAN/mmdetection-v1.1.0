'''
copied and modified version from mmdet deform_conv.py
'''

import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single
from mmdet.utils import print_log
from mmdet.ops.dcn import DeformConvPack, ModulatedDeformConvPack


deform_conv_cfg = {
    'DCN': DeformConvPack,
    'DCNv2': ModulatedDeformConvPack,
}

def build_deform_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        raise ValueError
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in deform_conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = deform_conv_cfg[layer_type]
    layer = conv_layer(*args, **kwargs, **cfg_)
    return layer
