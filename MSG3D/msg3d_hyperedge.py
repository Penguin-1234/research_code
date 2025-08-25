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

# ハイパーエッジを定義するクラス
class HyperedgeDefinition:
    """マイクロジェスチャーカテゴリに対応するハイパーエッジを定義"""
    
    def __init__(self, num_joints=25):
        self.num_joints = num_joints
        self.hyperedges = self._define_hyperedges()
    
    def _define_hyperedges(self):
        """NTU RGB+D 25関節モデルに基づくハイパーエッジ定義"""
        # NTU RGB+D 25関節のインデックス（0-24）
        # 0: base of spine, 1: middle of spine, 2: neck, 3: head
        # 4: left shoulder, 5: left elbow, 6: left wrist, 7: left hand
        # 8: right shoulder, 9: right elbow, 10: right wrist, 11: right hand
        # 12: left hip, 13: left knee, 14: left ankle, 15: left foot
        # 16: right hip, 17: right knee, 18: right ankle, 19: right foot
        # 20: spine, 21: left hand tip, 22: left thumb, 23: right hand tip, 24: right thumb
        
        hyperedges = {
            # whole body: 全身の主要関節
            'whole_body': [0, 1, 2, 3, 4, 8, 12, 16, 20],
            
            # hand-body: 手と胴体の関係
            'hand_body': [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24],
            
            # hand-head: 手と頭の関係
            'hand_head': [2, 3, 6, 7, 10, 11, 21, 22, 23, 24],
            
            # hand-hand: 両手の関係
            'hand_hand': [6, 7, 10, 11, 21, 22, 23, 24],
            
            # non-MG: 基本的な構造関節（最小限の関節セット）
            'non_mg': [0, 1, 2, 20]
        }
        
        return hyperedges
    
    def get_hyperedge_adjacency(self, hyperedge_type):
        """指定されたハイパーエッジタイプの隣接行列を生成"""
        joints = self.hyperedges[hyperedge_type]
        adj = torch.zeros(self.num_joints, self.num_joints)
        
        # ハイパーエッジ内の全関節間に接続を作成
        for i in joints:
            for j in joints:
                if i != j:
                    adj[i, j] = 1.0
        
        return adj

# ハイパーエッジ対応のグラフ畳み込み層
class HyperedgeGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints=25):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints
        
        # ハイパーエッジ定義
        self.hyperedge_def = HyperedgeDefinition(num_joints)
        
        # 各ハイパーエッジタイプに対応する畳み込み層
        self.hyperedge_convs = nn.ModuleDict({
            edge_type: nn.Conv2d(in_channels, out_channels, 1)
            for edge_type in self.hyperedge_def.hyperedges.keys()
        })
        
        # 重み付け学習用パラメータ
        self.hyperedge_weights = nn.Parameter(torch.ones(len(self.hyperedge_def.hyperedges)))
        
        # 正規化層
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, hyperedge_type_weights=None):
        """
        x: (N, C, T, V) - バッチサイズ、チャンネル、時間、関節数
        hyperedge_type_weights: 各ハイパーエッジタイプの重み（推論時に使用）
        """
        N, C, T, V = x.shape
        
        # 各ハイパーエッジからの特徴を計算
        hyperedge_features = []
        
        for i, (edge_type, conv) in enumerate(self.hyperedge_convs.items()):
            # ハイパーエッジの隣接行列を取得
            adj = self.hyperedge_def.get_hyperedge_adjacency(edge_type).to(x.device)
            
            # グラフ畳み込み適用
            # x: (N, C, T, V) -> (N*T, C, V)
            x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(N*T, C, V)
            
            # 隣接行列による重み付け集約
            # (N*T, C, V) × (V, V) -> (N*T, C, V)
            x_aggregated = torch.bmm(x_reshaped, adj.unsqueeze(0).expand(N*T, -1, -1))
            
            # 元の形状に戻す
            x_aggregated = x_aggregated.view(N, T, C, V).permute(0, 2, 1, 3)
            
            # 畳み込み適用
            feature = conv(x_aggregated)
            
            # 重み付け
            if hyperedge_type_weights is not None:
                weight = hyperedge_type_weights[i]
            else:
                weight = F.softmax(self.hyperedge_weights, dim=0)[i]
            
            hyperedge_features.append(feature * weight)
        
        # 全ハイパーエッジからの特徴を統合
        out = sum(hyperedge_features)
        out = self.bn(out)
        
        return out

# ハイパーエッジ対応のMS_G3D
class HyperedgeMS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride, # この引数を内部で利用します
                 window_dilation,
                 num_joints=25,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.window_stride = window_stride # strideを保存
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        # 元のMSG3D構造
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
        
        # ハイパーエッジ処理
        self.hyperedge_gcn = HyperedgeGraphConv(
            self.embed_channels_in, 
            self.embed_channels_out,
            num_joints
        )

        # ★★★ 修正箇所: stride > 1 の場合に使用するプーリング層を追加 ★★★
        if self.window_stride > 1:
            self.hyperedge_pool = nn.AvgPool2d(kernel_size=(self.window_stride, 1), stride=(self.window_stride, 1))
        else:
            self.hyperedge_pool = nn.Identity()


        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)
        
        # 統合重み
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, hyperedge_type_weights):
        N, C, T, V = x.shape
        x_in = self.in1x1(x)
        
        # 元のMSG3D特徴
        x_msg3d = self.gcn3d(x_in)
        x_msg3d = x_msg3d.view(N, self.embed_channels_out, -1, self.window_size, V)
        x_msg3d = self.out_conv(x_msg3d).squeeze(dim=3)
        
        # ハイパーエッジ特徴
        x_hyperedge = self.hyperedge_gcn(x_in, hyperedge_type_weights)
        
        # ★★★ 修正箇所: 時間軸をダウンサンプリングしてサイズを合わせる ★★★
        x_hyperedge = self.hyperedge_pool(x_hyperedge)
        
        # パディングなどにより、1フレームずれる可能性への対処
        T_msg3d = x_msg3d.shape[2]
        T_hyper = x_hyperedge.shape[2]
        if T_msg3d != T_hyper:
            # x_hyperedge の時間長を x_msg3d に合わせる
            x_hyperedge = x_hyperedge[:, :, :T_msg3d, :]
        
        # 特徴統合
        fusion_w = torch.sigmoid(self.fusion_weight)
        x_fused = fusion_w * x_msg3d + (1 - fusion_w) * x_hyperedge
        
        x_fused = self.out_bn(x_fused)
        
        return x_fused

# マルチウィンドウ対応のハイパーエッジMS_G3D
class MultiWindow_HyperedgeMS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3,5],
                 window_stride=1,
                 window_dilations=[1,1],
                 num_joints=25):

        super().__init__()
        self.gcn3d = nn.ModuleList([
            HyperedgeMS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation,
                num_joints
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x, hyperedge_type_weights=None):
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x, hyperedge_type_weights)
        return out_sum

# マイクロジェスチャー分類のためのハイパーエッジ対応モデル
class HyperedgeMSG3DModel(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3,
                 num_joints=25):
        super(HyperedgeMSG3DModel, self).__init__()
        
        self.num_class = num_class
        self.num_joints = num_joints

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = c2 * 2     # 384

        # ハイパーエッジ対応のGCN3Dブロック
        self.gcn3d1 = MultiWindow_HyperedgeMS_G3D(3, c1, A_binary, num_g3d_scales, 
                                                  window_stride=1, num_joints=num_joints)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = MultiWindow_HyperedgeMS_G3D(c1, c2, A_binary, num_g3d_scales, 
                                                  window_stride=2, num_joints=num_joints)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.gcn3d3 = MultiWindow_HyperedgeMS_G3D(c2, c3, A_binary, num_g3d_scales, 
                                                  window_stride=2, num_joints=num_joints)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        # マイクロジェスチャー分類用のヘッド
        self.classifier = nn.Sequential(
            nn.Linear(c3, c3 // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(c3 // 2, num_class)
        )
        
        # カテゴリ予測用の補助分類器（5カテゴリ：non-MG, whole body, hand-body, hand-head, hand-hand）
        self.category_classifier = nn.Sequential(
            nn.Linear(c3, c3 // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(c3 // 4, 5)
        )

    def forward(self, x, return_category_pred=False):
        # 前処理
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
        
        # カテゴリ予測から各ハイパーエッジタイプの重みを計算
        # 最初の層で簡単な特徴抽出
        x_temp = F.relu(self.sgcn1(x), inplace=True)
        x_temp = self.tcn1(x_temp)
        
        # 全体特徴からカテゴリ予測
        temp_pool = x_temp.view(N, M, x_temp.size(1), -1).mean(3).mean(1)
        category_logits = self.category_classifier(temp_pool)
        hyperedge_type_weights = F.softmax(category_logits, dim=1)
        
        # ハイパーエッジ重みを使用してモデルを通す
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x, hyperedge_type_weights), inplace=True)
        x = self.tcn1(x)
        
        x = F.relu(self.sgcn2(x) + self.gcn3d2(x, hyperedge_type_weights), inplace=True)
        x = self.tcn2(x)
        
        x = F.relu(self.sgcn3(x) + self.gcn3d3(x, hyperedge_type_weights), inplace=True)
        x = self.tcn3(x)

        # プーリングと分類
        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence

        # 最終分類
        class_logits = self.classifier(out)
        
        if return_category_pred:
            return class_logits, category_logits
        else:
            return class_logits

# マルチタスク学習用の損失関数
class MultiTaskLoss(nn.Module):
    def __init__(self, class_weight=1.0, category_weight=0.3):
        super().__init__()
        self.class_weight = class_weight
        self.category_weight = category_weight
        self.class_loss = nn.CrossEntropyLoss()
        self.category_loss = nn.CrossEntropyLoss()
    
    def forward(self, class_logits, category_logits, class_labels, category_labels):
        loss_class = self.class_loss(class_logits, class_labels)
        loss_category = self.category_loss(category_logits, category_labels)
        
        total_loss = (self.class_weight * loss_class + 
                     self.category_weight * loss_category)
        
        return total_loss, loss_class, loss_category

if __name__ == "__main__":
    # デバッグ用
    import sys
    sys.path.append('..')

    # マイクロジェスチャー分類用のモデル
    model = HyperedgeMSG3DModel(
        num_class=17,  # マイクロジェスチャークラス数
        num_point=25,
        num_person=1,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph',
        num_joints=25
    )

    # テストデータ
    N, C, T, V, M = 6, 3, 50, 25, 2
    x = torch.randn(N, C, T, V, M)
    
    # 推論テスト
    class_logits, category_logits = model(x, return_category_pred=True)
    
    print(f'Model total # params: {count_params(model)}')
    print(f'Class logits shape: {class_logits.shape}')
    print(f'Category logits shape: {category_logits.shape}')
    
    # 損失関数テスト
    class_labels = torch.randint(0, 17, (N,))
    category_labels = torch.randint(0, 5, (N,))
    
    loss_fn = MultiTaskLoss()
    total_loss, class_loss, category_loss = loss_fn(
        class_logits, category_logits, class_labels, category_labels
    )
    
    print(f'Total loss: {total_loss.item():.4f}')
    print(f'Class loss: {class_loss.item():.4f}')
    print(f'Category loss: {category_loss.item():.4f}')