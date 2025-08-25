# MSG3D/msg3d.py の修正版

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

class SIGNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=5):
        super().__init__()
        self.num_groups = num_groups
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc_out = nn.Linear(out_channels * 2, out_channels)  # 結合用

    def forward(self, x):
        # x: [N, G, C]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # [N, G, out]
        x = x.permute(0, 2, 1)  # [N, out, G]

        avg_pooled = self.avg_pool(x).squeeze(-1)  # [N, out]
        max_pooled = self.max_pool(x).squeeze(-1)  # [N, out]
        x = torch.cat([avg_pooled, max_pooled], dim=1)  # [N, out*2]
        x = self.fc_out(x)  # [N, out]

        return x


class MS_G3D_SIGNet(nn.Module):
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
            # The first STGC block changes channels right away; others change at collapse
            # ### 変更点: ハードコーディングされていた '3' を in_channels に変更 ###
            if in_channels == 6: # もしくは in_channels > 3 など、柔軟な条件にしても良い
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
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
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
            MS_G3D_SIGNet(
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
        # Input shape: (N, C, T, V)
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        # no activation
        return out_sum


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3): # デフォルトは3のままにしておく
        super(Model, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = c2 * 2     # 384

        self.gcn3d1 = MultiWindow_MS_G3D(in_channels, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, in_channels, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        self.sig_net=SIGNet(in_channels=c3,out_channels=128,num_groups=5)  # SIGNetの追加
        #self.fc = nn.Linear(c3+128, num_class)
        self.fc = nn.Linear(128, num_class)  # SIGNetの出力を直接使用

    def forward(self, x):
        # 前処理
        N, C, T, V, M = x.size()
        joint_x=x
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()

        # Apply activation to the sum of the pathways
        # 第一層の処理
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)
        # 第二層の処理
        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)
        # 第三層の処理
        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = self.tcn3(x)

        group_map={
            0:[4,5,6,7,21,22],#左腕
            1:[8,9,10,11,23,24],#右腕
            2:[12,13,14,15],#左足
            3:[16,17,18,19],#右足
            4:[0,1,2,20]#胴体
        }

        group_features=[]
        for joints in group_map.values():
            group_x=x[:,:,:,joints]
            #group_x = joint_x[:,:,:,joints]  # joint_xを使用
            group_x = group_x.mean(dim=3)
            group_x = group_x.mean(dim=2)
            group_features.append(group_x)
        
        sig_input=torch.stack(group_features, dim=1)  # [N, G, C]
        sig_output = self.sig_net(sig_input)  # [N, out]


        # # プーリングと分類
        # out = x
        # out_channels = out.size(1)
        # out = out.view(N, M, out_channels, -1)
        # out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        # out = out.mean(1)   # Average pool number of bodies in the sequence

        # sig_output=sig_output.view(N,M,-1).mean(1)

        # out=torch.cat((out, sig_output), dim=1)  # [N, C + out]
        out = self.fc(sig_output)
        return out


if __name__ == "__main__":
    # For debugging purposes
    import sys
    sys.path.append('..')

    model = Model(
        num_class=17,
        num_point=25,
        num_person=1,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph',
        in_channels=6 # 修正が正しく機能するかテスト
    )

    N, C, T, V, M = 6, 6, 50, 25, 2 # 入力チャンネルを6に
    x = torch.randn(N,C,T,V,M)
    model.forward(x) # エラーが出なければOK

    print('Model total # params:', count_params(model))