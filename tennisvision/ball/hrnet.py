"""HRNet model definition used by WASB-SBDT for ball tracking.

Verbatim (with minor comment cleanup) from nttcom/WASB-SBDT's
src/models/hrnet.py.  The upstream file carries the Microsoft MIT
license header reproduced below; the WASB-SBDT repo itself is MIT as
well, so the derived file is compatible.
"""

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import annotations

import logging

import torch.nn as nn

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion,
                                     momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


_BLOCKS = {"BASIC": _BasicBlock, "BOTTLENECK": _Bottleneck}


class _HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super().__init__()
        if num_branches != len(num_blocks):
            raise ValueError("NUM_BRANCHES != NUM_BLOCKS")
        if num_branches != len(num_channels):
            raise ValueError("NUM_BRANCHES != NUM_CHANNELS")
        if num_branches != len(num_inchannels):
            raise ValueError("NUM_BRANCHES != NUM_INCHANNELS")

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if (stride != 1 or
                self.num_inchannels[branch_index] !=
                num_channels[branch_index] * block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )
        layers = [block(self.num_inchannels[branch_index],
                        num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = (
            num_channels[branch_index] * block.expansion)
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([
            self._make_one_branch(i, block, num_blocks, num_channels)
            for i in range(num_branches)
        ])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i],
                                  1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            out_c = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], out_c,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(out_c),
                            ))
                        else:
                            out_c = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], out_c,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU(True),
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNet(nn.Module):
    """WASB's HRNet variant.  `cfg` is a dict (any mapping) that
    supports either bracket OR attribute access; see AttrDict helper
    below — only `_make_deconv_layers` needs attribute access, and only
    if NUM_DECONVS > 0 (which tennis config doesn't).  For safety we
    coerce everything to bracket access."""

    def __init__(self, cfg, **_):
        super().__init__()

        self._frames_in  = cfg["frames_in"]
        self._frames_out = cfg["frames_out"]
        self._out_scales = cfg["out_scales"]
        extra = cfg["MODEL"]["EXTRA"]
        stem_strides  = extra["STEM"]["STRIDES"]
        stem_inplanes = extra["STEM"]["INPLANES"]

        self.conv1 = nn.Conv2d(3 * self._frames_in, stem_inplanes,
                               kernel_size=3, stride=stem_strides[0],
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(stem_inplanes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(stem_inplanes, stem_inplanes, kernel_size=3,
                               stride=stem_strides[1], padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(stem_inplanes, momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU(inplace=True)

        self.stage1_cfg = extra["STAGE1"]
        block = _BLOCKS[self.stage1_cfg["BLOCK"]]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0]
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]
        self.layer1 = self._make_layer(block, stem_inplanes, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = extra["STAGE2"]
        self.transition1, self.stage2, pre_channels = self._build_stage(
            [stage1_out_channel], self.stage2_cfg)

        self.stage3_cfg = extra["STAGE3"]
        self.transition2, self.stage3, pre_channels = self._build_stage(
            pre_channels, self.stage3_cfg)

        self.stage4_cfg = extra["STAGE4"]
        self.transition3, self.stage4, pre_channels = self._build_stage(
            pre_channels, self.stage4_cfg, multi_scale_output=True)

        self.num_deconvs = extra["DECONV"]["NUM_DECONVS"]
        self.deconv_layers = nn.ModuleList()   # empty for tennis (NUM_DECONVS=0)
        self.final_layers = self._make_final_layers(extra, pre_channels)

    # --- helpers ---

    def _build_stage(self, pre_channels, stage_cfg, multi_scale_output=True):
        block = _BLOCKS[stage_cfg["BLOCK"]]
        num_channels = [c * block.expansion for c in stage_cfg["NUM_CHANNELS"]]
        transition = self._make_transition_layer(pre_channels, num_channels)
        stage, pre = self._make_stage(stage_cfg, num_channels,
                                      multi_scale_output=multi_scale_output)
        return transition, stage, pre

    def _make_final_layers(self, extra, channels):
        k = extra["FINAL_CONV_KERNEL"]
        layers = [nn.Conv2d(in_channels=channels[scale],
                            out_channels=self._frames_out,
                            kernel_size=k)
                  for scale in self._out_scales]
        return nn.ModuleList(layers)

    def _make_transition_layer(self, pre, cur):
        num_branches_cur = len(cur)
        num_branches_pre = len(pre)
        layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if cur[i] != pre[i]:
                    layers.append(nn.Sequential(
                        nn.Conv2d(pre[i], cur[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(cur[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    layers.append(None)
            else:
                seq = []
                for j in range(i + 1 - num_branches_pre):
                    inc  = pre[-1]
                    outc = cur[i] if j == i - num_branches_pre else inc
                    seq.append(nn.Sequential(
                        nn.Conv2d(inc, outc, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outc, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                layers.append(nn.Sequential(*seq))
        return nn.ModuleList(layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_cfg, num_inchannels, multi_scale_output=True):
        num_modules = layer_cfg["NUM_MODULES"]
        num_branches = layer_cfg["NUM_BRANCHES"]
        num_blocks = layer_cfg["NUM_BLOCKS"]
        num_channels = layer_cfg["NUM_CHANNELS"]
        block = _BLOCKS[layer_cfg["BLOCK"]]
        fuse_method = layer_cfg["FUSE_METHOD"]
        modules = []
        for i in range(num_modules):
            reset_mso = not (not multi_scale_output and i == num_modules - 1)
            modules.append(_HighResolutionModule(
                num_branches, block, num_blocks, num_inchannels,
                num_channels, fuse_method, reset_mso,
            ))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        y_out = {}
        for scale in self._out_scales:
            feat = y_list[scale]
            # NOTE: tennis config uses NUM_DECONVS=0, so deconv_layers empty
            y_out[scale] = self.final_layers[scale](feat)
        return y_out


# --- model-config dict matching WASB's configs/model/wasb.yaml ---

WASB_CONFIG = {
    "frames_in":  3,
    "frames_out": 3,
    "out_scales": [0],
    "inp_height": 288,
    "inp_width":  512,
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "PRETRAINED_LAYERS": ["*"],
            "STEM":  {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK",
                       "NUM_BLOCKS": [1], "NUM_CHANNELS": [32], "FUSE_METHOD": "SUM"},
            "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32],
                       "FUSE_METHOD": "SUM"},
            "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64],
                       "FUSE_METHOD": "SUM"},
            "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [2, 2, 2, 2],
                       "NUM_CHANNELS": [16, 32, 64, 128],
                       "FUSE_METHOD": "SUM"},
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [],
                       "NUM_BASIC_BLOCKS": 2},
        },
        "INIT_WEIGHTS": True,
    },
}
