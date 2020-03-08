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
# from mmdet.ops.dcn.deform_conv import deform_conv, modulated_deform_conv


# class DeformConv(nn.Module):

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  groups=1,
#                  deformable_groups=1,
#                  bias=False,
#                  name=None):
#         super(DeformConv, self).__init__()

#         assert not bias
#         assert in_channels % groups == 0, \
#             'in_channels {} cannot be divisible by groups {}'.format(
#                 in_channels, groups)
#         assert out_channels % groups == 0, \
#             'out_channels {} cannot be divisible by groups {}'.format(
#                 out_channels, groups)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#         self.groups = groups
#         self.deformable_groups = deformable_groups
#         # enable compatibility with nn.Conv2d
#         self.transposed = False
#         self.output_padding = _single(0)

#         # self.weight = nn.Parameter(
#         #     torch.Tensor(out_channels, in_channels // self.groups,
#         #                  *self.kernel_size))
#         self.weight = nn.Parameter(
#             torch.Tensor(out_channels, in_channels // self.groups,
#                          *self.kernel_size))
#         assert name is not None
#         # self.register_parameter(name, self.weight)  # v-qiaofl added  TODO check
#         self.reset_parameters()

#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)

#     def forward(self, x, offset):
#         return deform_conv(x, offset, self.weight, self.stride, self.padding,
#                            self.dilation, self.groups, self.deformable_groups)


# class ModulatedDeformConv(nn.Module):

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  groups=1,
#                  deformable_groups=1,
#                  bias=True,
#                  name=None):
#         super(ModulatedDeformConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.deformable_groups = deformable_groups
#         self.with_bias = bias
#         # enable compatibility with nn.Conv2d
#         self.transposed = False
#         self.output_padding = _single(0)

#         self.weight = nn.Parameter(
#             torch.Tensor(out_channels, in_channels // groups,
#                          *self.kernel_size))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#         assert name is not None
#         # self.register_parameter(name, self.weight)  # v-qiaofl added, TODO check
#         self.reset_parameters()

#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.zero_()

#     def forward(self, x, offset, mask):
#         return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
#                                      self.stride, self.padding, self.dilation,
#                                      self.groups, self.deformable_groups)


# class DeformConvPack(DeformConv):
#     """A Deformable Conv Encapsulation that acts as normal Conv layers.

#     Args:
#         in_channels (int): Same as nn.Conv2d.
#         out_channels (int): Same as nn.Conv2d.
#         kernel_size (int or tuple[int]): Same as nn.Conv2d.
#         stride (int or tuple[int]): Same as nn.Conv2d.
#         padding (int or tuple[int]): Same as nn.Conv2d.
#         dilation (int or tuple[int]): Same as nn.Conv2d.
#         groups (int): Same as nn.Conv2d.
#         bias (bool or str): If specified as `auto`, it will be decided by the
#             norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
#             False.
#     """

#     _version = 2

#     def __init__(self, *args, **kwargs):
#         super(DeformConvPack, self).__init__(*args, **kwargs)

#         self.conv_offset = nn.Conv2d(
#             self.in_channels,
#             self.deformable_groups * 2 * self.kernel_size[0] *
#             self.kernel_size[1],
#             kernel_size=self.kernel_size,
#             stride=_pair(self.stride),
#             padding=_pair(self.padding),
#             bias=True)
#         self.init_offset()

#     def init_offset(self):
#         self.conv_offset.weight.data.zero_()
#         self.conv_offset.bias.data.zero_()

#     def forward(self, x):
#         offset = self.conv_offset(x)
#         return deform_conv(x, offset, self.weight, self.stride, self.padding,
#                            self.dilation, self.groups, self.deformable_groups)

#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         version = local_metadata.get('version', None)

#         if version is None or version < 2:
#             # the key is different in early versions
#             # In version < 2, DeformConvPack loads previous benchmark models.
#             if (prefix + 'conv_offset.weight' not in state_dict
#                     and prefix[:-1] + '_offset.weight' in state_dict):
#                 state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
#                     prefix[:-1] + '_offset.weight')
#             if (prefix + 'conv_offset.bias' not in state_dict
#                     and prefix[:-1] + '_offset.bias' in state_dict):
#                 state_dict[prefix +
#                            'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
#                                                                 '_offset.bias')

#         if version is not None and version > 1:
#             print_log(
#                 'DeformConvPack {} is upgraded to version 2.'.format(
#                     prefix.rstrip('.')),
#                 logger='root')

#         super()._load_from_state_dict(state_dict, prefix, local_metadata,
#                                       strict, missing_keys, unexpected_keys,
#                                       error_msgs)


# class ModulatedDeformConvPack(ModulatedDeformConv):
#     """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

#     Args:
#         in_channels (int): Same as nn.Conv2d.
#         out_channels (int): Same as nn.Conv2d.
#         kernel_size (int or tuple[int]): Same as nn.Conv2d.
#         stride (int or tuple[int]): Same as nn.Conv2d.
#         padding (int or tuple[int]): Same as nn.Conv2d.
#         dilation (int or tuple[int]): Same as nn.Conv2d.
#         groups (int): Same as nn.Conv2d.
#         bias (bool or str): If specified as `auto`, it will be decided by the
#             norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
#             False.
#     """

#     _version = 2

#     def __init__(self, *args, **kwargs):
#         super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

#         self.conv_offset = nn.Conv2d(
#             self.in_channels,
#             self.deformable_groups * 3 * self.kernel_size[0] *
#             self.kernel_size[1],
#             kernel_size=self.kernel_size,
#             stride=_pair(self.stride),
#             padding=_pair(self.padding),
#             bias=True)
#         self.init_offset()

#     def init_offset(self):
#         self.conv_offset.weight.data.zero_()
#         self.conv_offset.bias.data.zero_()

#     def forward(self, x):
#         out = self.conv_offset(x)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#         return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
#                                      self.stride, self.padding, self.dilation,
#                                      self.groups, self.deformable_groups)

#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         version = local_metadata.get('version', None)

#         if version is None or version < 2:
#             # the key is different in early versions
#             # In version < 2, ModulatedDeformConvPack
#             # loads previous benchmark models.
#             if (prefix + 'conv_offset.weight' not in state_dict
#                     and prefix[:-1] + '_offset.weight' in state_dict):
#                 state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
#                     prefix[:-1] + '_offset.weight')
#             if (prefix + 'conv_offset.bias' not in state_dict
#                     and prefix[:-1] + '_offset.bias' in state_dict):
#                 state_dict[prefix +
#                            'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
#                                                                 '_offset.bias')

#         if version is not None and version > 1:
#             print_log(
#                 'ModulatedDeformConvPack {} is upgraded to version 2.'.format(
#                     prefix.rstrip('.')),
#                 logger='root')

#         super()._load_from_state_dict(state_dict, prefix, local_metadata,
#                                       strict, missing_keys, unexpected_keys,
#                                       error_msgs)



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
