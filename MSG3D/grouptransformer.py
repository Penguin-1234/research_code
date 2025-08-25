# MSG3D/msg3d.py の修正版 - Transformerベースのグループ相互作用

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


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class GroupTransformerNet(nn.Module):
    """Transformer-based network for capturing group interactions"""
    def __init__(self, in_channels, hidden_dim=256, num_heads=8, num_layers=3, dropout=0.1, num_groups=5):
        super().__init__()
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # Positional encoding for groups
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=num_groups)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Group-specific attention weights
        self.group_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers with residual connection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Global aggregation with learnable weights
        self.global_weights = nn.Parameter(torch.ones(num_groups) / num_groups)
        self.final_proj = nn.Linear(hidden_dim // 4, 128)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [N, G, C] where G is number of groups
        batch_size, num_groups, in_channels = x.size()
        
        # Project to hidden dimension
        x = self.input_proj(x)  # [N, G, hidden_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.layer_norm1(x)
        
        # Apply transformer encoder
        transformer_out = self.transformer(x)  # [N, G, hidden_dim]
        
        # Apply group-specific attention
        attn_out, attn_weights = self.group_attention(
            transformer_out, transformer_out, transformer_out
        )
        
        # Residual connection
        x = self.layer_norm2(transformer_out + attn_out)
        x = self.dropout(x)
        
        # Project to output dimension
        group_features = self.output_proj(x)  # [N, G, hidden_dim//4]
        
        # Weighted global aggregation
        weights = F.softmax(self.global_weights, dim=0)
        global_feature = torch.sum(group_features * weights.unsqueeze(0).unsqueeze(-1), dim=1)  # [N, hidden_dim//4]
        
        # Final projection
        output = self.final_proj(global_feature)  # [N, 128]
        
        return output, attn_weights  # Return attention weights for interpretability


class MS_G3D_GTformer(nn.Module):
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
            if in_channels == 6:
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
            MS_G3D_GTformer(
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


# class Model(nn.Module):
#     def __init__(self,
#                  num_class,
#                  num_point,
#                  num_person,
#                  num_gcn_scales,
#                  num_g3d_scales,
#                  graph,
#                  in_channels=3,
#                  transformer_hidden_dim=256,
#                  transformer_heads=8,
#                  transformer_layers=3,
#                  transformer_dropout=0.1):
#         super(Model, self).__init__()

#         Graph = import_class(graph)
#         A_binary = Graph().A_binary

#         self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

#         # channels
#         c1 = 96
#         c2 = c1 * 2
#         c3 = c2 * 2

#         self.gcn3d1 = MultiWindow_MS_G3D(in_channels, c1, A_binary, num_g3d_scales, window_stride=1)
#         self.sgcn1 = nn.Sequential(
#             MS_GCN(num_gcn_scales, in_channels, c1, A_binary, disentangled_agg=True),
#             MS_TCN(c1, c1),
#             MS_TCN(c1, c1))
#         self.sgcn1[-1].act = nn.Identity()
#         self.tcn1 = MS_TCN(c1, c1)

#         self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
#         self.sgcn2 = nn.Sequential(
#             MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
#             MS_TCN(c1, c2, stride=2),
#             MS_TCN(c2, c2))
#         self.sgcn2[-1].act = nn.Identity()
#         self.tcn2 = MS_TCN(c2, c2)

#         self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
#         self.sgcn3 = nn.Sequential(
#             MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
#             MS_TCN(c2, c3, stride=2),
#             MS_TCN(c3, c3))
#         self.sgcn3[-1].act = nn.Identity()
#         self.tcn3 = MS_TCN(c3, c3)

#         self.group_transformer = GroupTransformerNet(
#             in_channels=c3,
#             hidden_dim=transformer_hidden_dim,
#             num_heads=transformer_heads,
#             num_layers=transformer_layers,
#             dropout=transformer_dropout,
#             num_groups=5
#         )

#         self.feature_fusion = nn.Sequential(
#             nn.Linear(c3 + 128, (c3 + 128) // 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear((c3 + 128) // 2, (c3 + 128) // 4)
#         )

#         self.fc = nn.Linear((c3 + 128) // 4, num_class)

#         # 固定グループの定義（例: [右腕, 左腕, 頭, 体幹, 脚]）
#         self.group_map = {
#             0: [4, 5, 6, 7],       # 右腕
#             1: [8, 9, 10, 11],    # 左腕
#             2: [0, 1, 2, 3],      # 頭・首
#             3: [12, 13, 14],      # 胴体
#             4: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # 両脚
#         }

#     def get_group_features_fixed(self, x):
#         N, C, T, V = x.shape
#         group_features = []
#         for joints in self.group_map.values():
#             group_x = x[:, :, :, joints]        # [N, C, T, J']
#             group_x = group_x.mean(dim=3)       # [N, C, T]
#             group_x = group_x.mean(dim=2)       # [N, C]
#             group_features.append(group_x)
#         return torch.stack(group_features, dim=1)  # [N, G, C]

#     def forward(self, x):
#         N, C, T, V, M = x.size()
#         x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
#         x = self.data_bn(x)
#         x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

#         x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
#         x = self.tcn1(x)
#         x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
#         x = self.tcn2(x)
#         x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
#         x = self.tcn3(x)

#         sig_input = self.get_group_features_fixed(x)  # [N*M, G, C]
#         sig_output, attention_weights = self.group_transformer(sig_input)  # [N*M, 128]

#         out = x
#         out_channels = out.size(1)
#         out = out.view(N, M, out_channels, -1)
#         out = out.mean(3)
#         out = out.mean(1)

#         sig_output = sig_output.view(N, M, -1).mean(1)

#         combined_features = torch.cat((out, sig_output), dim=1)
#         fused_features = self.feature_fusion(combined_features)
#         out = self.fc(fused_features)
#         return out

class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3,
                 transformer_hidden_dim=256,
                 transformer_heads=8,
                 transformer_layers=3,
                 transformer_dropout=0.1):
        super(Model, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        c1 = 96
        c2 = c1 * 2
        c3 = c2 * 2

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

        self.group_map = {
            0: [4, 5, 6, 7],        # 右腕
            1: [8, 9, 10, 11],     # 左腕
            2: [0, 1, 2, 3],       # 頭・首
            3: [12, 13, 14],       # 胴体
            4: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # 両脚
        }

        self.group_transformer = GroupTransformerNet(
            in_channels=c3,
            hidden_dim=transformer_hidden_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=transformer_dropout,
            num_groups=5
        )

        self.classifier = nn.Sequential(
            nn.Linear(transformer_hidden_dim//2, transformer_hidden_dim // 2),
            #nn.Linear(transformer_hidden_dim, transformer_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(transformer_hidden_dim // 2, num_class)
        )

    def get_group_features_fixed(self, x):
        N, C, T, V = x.shape
        group_features = []
        for joints in self.group_map.values():
            group_x = x[:, :, :, joints]        # [N, C, T, J']
            group_x = group_x.mean(dim=3)       # [N, C, T]
            group_x = group_x.mean(dim=2)       # [N, C]
            group_features.append(group_x)
        return torch.stack(group_features, dim=1)  # [N, G, C]

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

        sig_input = self.get_group_features_fixed(x)  # [N*M, G, C]
        sig_output, _ = self.group_transformer(sig_input)  # [N*M, 128]

        out = sig_output.view(N, M, -1).mean(1)  # 平均化
        out = self.classifier(out)  # 分類器に通す
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
        in_channels=6,
        transformer_hidden_dim=256,
        transformer_heads=8,
        transformer_layers=3,
        transformer_dropout=0.1
    )

    N, C, T, V, M = 6, 6, 50, 25, 2
    x = torch.randn(N,C,T,V,M)
    output = model.forward(x)
    
    print('Model total # params:', count_params(model))
    print('Output shape:', output.shape)