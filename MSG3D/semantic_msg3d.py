import sys
sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params
from MSG3D.ms_gcn import MultiScale_GraphConv as MS_GCN
from MSG3D.ms_tcn import MultiScale_TemporalConv as MS_TCN
from MSG3D.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from MSG3D.mlp import MLP
from MSG3D.activation import activation_factory

# ------------------- Semantic Group Definition -------------------
group_definitions = {
    "whole_body": list(range(25)),
    "hand_body": [4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 18],
    "hand_head": [4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 18],
    "hand_hand": [4, 5, 6, 7, 8, 9, 10, 11, 19, 20, 21, 22],
    "non_micro": list(range(25))
}

# ------------------- Modules -------------------

class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )
        )

        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        x = self.gcn3d(x)
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)
        return x

class MultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3,5],
                 window_stride=1,
                 window_dilations=[1,1]):
        super().__init__()
        self.gcn3d = nn.ModuleList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        return out_sum

class SemanticMultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_A_binary,
                 num_scales,
                 group_definitions,
                 window_sizes=[3,5],
                 window_stride=1,
                 window_dilations=[1,1]):
        super().__init__()
        self.group_names = list(group_definitions.keys())
        self.group_indices = group_definitions

        self.group_modules = nn.ModuleDict()
        for name in self.group_names:
            node_ids = group_definitions[name]
            sub_A = base_A_binary[np.ix_(node_ids, node_ids)]
            self.group_modules[name] = MultiWindow_MS_G3D(
                in_channels,
                out_channels,
                sub_A,
                num_scales,
                window_sizes,
                window_stride,
                window_dilations
            )

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * len(self.group_names), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        N, C, T, V = x.shape
        group_outputs = []
        for name in self.group_names:
            idx = self.group_indices[name]
            x_sub = x[:, :, :, idx]
            out = self.group_modules[name](x_sub)
            group_outputs.append(out)
        out = torch.cat(group_outputs, dim=1)
        out = self.fuse(out)
        return out

# ------------------- Model -------------------

class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(Model, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        c1 = 96
        c2 = c1 * 2
        c3 = c2 * 2

        self.gcn3d1 = SemanticMultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, group_definitions)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = SemanticMultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, group_definitions)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.gcn3d3 = SemanticMultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, group_definitions)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        self.fc = nn.Linear(c3, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)

        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = self.tcn3(x)

        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)
        out = out.mean(1)

        out = self.fc(out)
        return out
