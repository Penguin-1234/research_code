# graph/smg_30_points.py

import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np
from graph import tools

# --- 30点モデルの定義 ---
# 関節点数
num_node = 30

# 新しい5点のインデックス (0-based)
# 25: 胴体 (Torso)
# 26: 右腕 (Right Arm)
# 27: 左腕 (Left Arm)
# 28: 右脚 (Right Leg)
# 29: 左脚 (Left Leg)
# ※インデックスはプログラム上では0から始まるため、25から29となります。

# 自己ループ (0-based)
self_link = [(i, i) for i in range(num_node)]

# 元の25関節の接続 (NTU-RGB+D由来、1-based index)
# 親から子への有向エッジとして定義
inward_ori_index_25 = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
    (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8),
    (24, 25), (25, 12)
]

# 新しい5点の接続 (1-based index)
# 1-25が元の関節、26-30が新しい代表点
inward_ori_index_5 = [
    # 26(胴体) <-> 体幹の主要部 (肩、腰、背骨)
    (21, 26), (5, 26), (9, 26), (13, 26), (17, 26), (1, 26),

    # 27(右腕代表) <-> 右腕の関節群 & 胴体
    (5, 27), (6, 27), (7, 27), (26, 27),

    # 28(左腕代表) <-> 左腕の関節群 & 胴体
    (9, 28), (10, 28), (11, 28), (26, 28),

    # 29(右脚代表) <-> 右脚の関節群 & 胴体
    (13, 29), (14, 29), (15, 29), (26, 29),

    # 30(左脚代表) <-> 左脚の関節群 & 胴体
    (17, 30), (18, 30), (19, 30), (26, 30),
]

# 全ての接続を結合
inward_ori_index = inward_ori_index_25 + inward_ori_index_5

# 1-basedを0-basedに変換
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    """
    30点の関節点とそれらの接続を定義するグラフクラス。
    MSG3Dモデルが必要とする多重スケールの隣接行列を生成します。
    """
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = self_link
        self.inward = inward
        self.outward = outward

        # 互換性のための単一スケール行列 (デバッグや可視化用)
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        
        # MSG3D用の多重スケール隣接行列
        self.A = self.get_A(*args, **kwargs)

    def get_A(self, strategy='spatial', num_gcn_scales=13):
        """
        MSG3Dで使われる多重スケールの隣接行列を生成します。
        'spatial'戦略に基づき、物理的な接続、ホップ数、方向性などを考慮します。
        """
        A = np.zeros((num_gcn_scales, self.num_nodes, self.num_nodes))
        A_binary_neighbor = tools.get_adjacency_matrix(self.edges, self.num_nodes)

        if strategy == 'spatial':
            # スケール0: 単位行列 (自己ループ)
            A[0] = tools.get_adjacency_matrix(self.self_loops, self.num_nodes)

            # スケール1: 正規化された隣接行列 (物理的な接続)
            A[1] = tools.normalize_adjacency_matrix(A_binary_neighbor)
            
            # スケール2: 2ホップ隣接行列
            A_binary_2hop = (self.A_binary @ self.A_binary > 0).astype(float)
            # 1ホップと自己ループは除く
            A_binary_2hop = A_binary_2hop - self.A_binary - A[0]
            A_binary_2hop[A_binary_2hop < 0] = 0
            A[2] = tools.normalize_adjacency_matrix(A_binary_2hop)

            # スケール3: 3ホップ隣接行列
            A_binary_3hop = (self.A_binary @ self.A_binary @ self.A_binary > 0).astype(float)
            # 1,2ホップと自己ループは除く
            A_binary_3hop = A_binary_3hop - A_binary_2hop - self.A_binary - A[0]
            A_binary_3hop[A_binary_3hop < 0] = 0
            A[3] = tools.normalize_adjacency_matrix(A_binary_3hop)

            # スケール4: inward (親->子)
            A[4] = tools.normalize_adjacency_matrix(tools.get_adjacency_matrix(self.inward, self.num_nodes))
            # スケール5: outward (子->親)
            A[5] = tools.normalize_adjacency_matrix(tools.get_adjacency_matrix(self.outward, self.num_nodes))

            # 残りのスケールを埋める (ここでは単純に物理接続をコピー)
            for i in range(6, num_gcn_scales):
                A[i] = np.copy(A[1]) 

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # グラフオブジェクトを生成
    graph = AdjMatrixGraph()
    A_multi_scale = graph.A
    A_binary = graph.A_binary
    A_binary_with_I = tools.get_adjacency_matrix(graph.edges + graph.self_loops, num_node)

    print("Shape of A (multi-scale for MSG3D):", A_multi_scale.shape)
    print("Shape of A_binary (for visualization):", A_binary.shape)

    # 可視化
    f, ax = plt.subplots(2, 3, figsize=(15, 10))
    plt.suptitle('SMG 30 Points Graph Adjacency Matrices', fontsize=16)

    ax[0, 0].imshow(A_binary_with_I, cmap='gray')
    ax[0, 0].set_title('A_binary_with_I (Neighbor + Self-loop)')

    ax[0, 1].imshow(A_binary, cmap='gray')
    ax[0, 1].set_title('A_binary (Neighbor)')
    
    ax[0, 2].imshow(tools.get_adjacency_matrix(inward, num_node), cmap='gray')
    ax[0, 2].set_title('Inward Edges Only')

    # 多重スケール行列の最初のいくつかを表示
    ax[1, 0].imshow(A_multi_scale[0], cmap='gray')
    ax[1, 0].set_title('MSG3D Scale 0 (Self-loop)')

    ax[1, 1].imshow(A_multi_scale[1], cmap='gray')
    ax[1, 1].set_title('MSG3D Scale 1 (Normalized Neighbor)')
    
    ax[1, 2].imshow(A_multi_scale[2], cmap='gray')
    ax[1, 2].set_title('MSG3D Scale 2 (2-hop Neighbor)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()