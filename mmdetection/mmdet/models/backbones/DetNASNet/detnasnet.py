import torch.nn as nn
from .shuffle_blocks import ConvBNReLU, ShuffleNetV2BlockSearched, blocks_key
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, kaiming_init
from ...registry import BACKBONES


@BACKBONES.register_module
class ShuffleNetV2DetNAS(nn.Module):
    def __init__(self, model_size, out_indices=(0, 1, 2, 3), frozen_stages=-1,norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False)):
        super(ShuffleNetV2DetNAS, self).__init__()
        # model_size = cfg.MODEL.BACKBONE.CONV_BODY.lstrip('DETNAS-')
        print('Model size is {}.'.format(model_size))

        if 'COCO-FPN-3.8G' in model_size:
            architecture = [0, 0, 3, 1, 2, 1, 0, 2, 0, 3, 1, 2, 3, 3, 2, 0, 2, 1, 1, 3,
                            2, 0, 2, 2, 2, 1, 3, 1, 0, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3]
            stage_repeats = [8, 8, 16, 8]
            stage_out_channels = [-1, 72, 172, 432, 864, 1728, 1728]
        elif 'COCO-FPN-1.3G' in model_size:
            architecture = [0, 0, 3, 1, 2, 1, 0, 2, 0, 3, 1, 2, 3, 3, 2, 0, 2, 1, 1, 3,
                            2, 0, 2, 2, 2, 1, 3, 1, 0, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3]
            stage_repeats = [8, 8, 16, 8]
            stage_out_channels = [-1, 48, 96, 240, 480, 960, 1024]
        elif 'COCO-FPN-300M' in model_size:
            architecture = [2, 1, 2, 0, 2, 1, 1, 2, 3, 3, 1, 3, 0, 0, 3, 1, 3, 1, 3, 2]
            #architecture = [0, 0, 0, 1, 2, 0, 3, 3, 1, 2, 2, 2, 3, 3, 3, 1, 3, 2, 3, 2] # search from tong 1019
            stage_repeats = [4, 4, 8, 4]
            stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        elif 'COCO-RetinaNet-300M' in model_size:
            architecture = [2, 3, 1, 1, 3, 2, 1, 3, 3, 1, 1, 1, 3, 3, 2, 0, 3, 3, 3, 3]
            stage_repeats = [4, 4, 8, 4]
            stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]  # NOTE last channel is only used for fc in imgnet classification task
        else:
            raise NotImplementedError

        if 'search' in model_size:
            architecture = None
            self.blocks_key = blocks_key
            self.num_states = sum(stage_repeats)
            raise ValueError # v-qiaofl added.

        self.first_conv = ConvBNReLU(in_channel=3, out_channel=stage_out_channels[1], k_size=3, stride=2, padding=1, gaussian_init=True)

        self.features = list()
        self.stage_ends_idx = list()
        self.norm_eval=norm_eval

        in_channels = stage_out_channels[1]
        i_th = 0
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
            assert fallback_on_stride is False
            assert len(stage_with_dcn) == len(out_indices)
            deformable_groups = dcn.pop('deformable_groups', 1)
            assert deformable_groups == 1
        self.gcb = gcb
        self.stage_with_gcb = stage_with_gcb
        if gcb is not None:
            assert len(stage_with_gcb) == len(out_indices)

        for id_stage in range(1, len(stage_repeats) + 1):
            out_channels = stage_out_channels[id_stage + 1]
            repeats = stage_repeats[id_stage - 1]
            dcn = self.dcn if self.stage_with_dcn[id_stage-1] else None
            gcb = self.gcb if self.stage_with_gcb[id_stage-1] else None
            for id_repeat in range(repeats):
                prefix = str(id_stage) + chr(ord('a') + id_repeat)
                stride = 1 if id_repeat > 0 else 2
                if architecture is None:
                    _ops = nn.ModuleList()
                    for i in range(len(blocks_key)):
                        _ops.append(ShuffleNetV2BlockSearched(prefix, in_channels=in_channels, out_channels=out_channels,
                                                               stride=stride, base_mid_channels=out_channels // 2, id=i))
                    self.features.append(_ops)
                    raise ValueError  # v-qiaofl added.
                else:
                    self.features.append(ShuffleNetV2BlockSearched(prefix, in_channels=in_channels, out_channels=out_channels,
                                                               stride=stride, base_mid_channels=out_channels // 2, id=architecture[i_th], dcn=dcn, gcb=gcb))
                in_channels = out_channels
                i_th += 1
            self.stage_ends_idx.append(i_th-1)
        assert len(self.stage_ends_idx) == len(out_indices)

        self.features = nn.Sequential(*self.features)
        self.frozen_stages = frozen_stages
        # if self.frozen_stages >= 0:
        #     raise NotImplementedError
        self._freeze_stages()
        self.feat_dim = stage_out_channels[-2]  # NOTE stage_out_channels[-1] is only used for fc in imgnet classification task
        

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
            # if self.dcn is not None:
            #     for m in self.modules():
            #         if isinstance(m, Bottleneck) and hasattr(
            #                 m, 'conv2_offset'):
            #             constant_init(m.conv2_offset, 0)

            # if self.zero_init_residual:
            #     for m in self.modules():
            #         if isinstance(m, Bottleneck):
            #             constant_init(m.norm3, 0)
            #         elif isinstance(m, BasicBlock):
            #             constant_init(m.norm2, 0)
            raise ValueError  # v-qiaofl added.
        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.first_conv.eval()
            for param in self.first_conv.parameters():
                param.requires_grad = False

        # NOTE TODO continue
        if self.frozen_stages >=1:
            end_idx = self.stage_ends_idx[self.frozen_stages-1]
            for i in range(end_idx+1):
                # print('self.features[i]: ', self.features[i])
                self.features[i].eval()
                for param in self.features[i].parameters():
                    param.requires_grad = False

    def forward(self, x, rngs=None):
        outputs = []
        x = self.first_conv(x)
        for i, select_op in enumerate(self.features):
            # x = select_op(x) if rngs is None else select_op[rngs[i]](x)  # v-qiaofl changed
            x = select_op(x)
            if i in self.stage_ends_idx:
                outputs.append(x)
        return outputs

    def train(self, mode=True):
        super(ShuffleNetV2DetNAS, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # print('m: ', m)
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


# if __name__ == "__main__":
#     from maskrcnn_benchmark.config import cfg
#     model = ShuffleNetV2DetNAS(cfg)
#     print(model)
