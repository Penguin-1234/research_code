import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pickle
import os
import time
import sys
from MSG3D.msg3d import Model as MSG3D_Model # Import the model from msg3d.py
from net.utils.graph import Graph
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import re

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        if self.alpha is not None:
            at = self.alpha.gather(0, target)
            logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

#x座標の反転
def horizontal_flip_skeleton(data_numpy):
    left_right_pairs = [
        (4, 8), (5, 7), (6, 10), (11, 17), (12, 16), (13, 15), (23, 21), (24, 22)
    ]
    data_flipped = np.copy(data_numpy)
    data_flipped[0, :, :, :] *= -1  # x座標反転
    for left, right in left_right_pairs:
        data_flipped[:, :, [left, right], :] = data_flipped[:, :, [right, left], :]
    return data_flipped

def temporal_reverse_skeleton(data_numpy):
    """
    時系列（T次元）を逆順にする
    data_numpy: (C, T, V, M)
    """
    return data_numpy[:, ::-1, :, :].copy()

class BaseSkeletonDataset(Dataset):
    """
    学習用・検証用データセットの共通処理をまとめた基底クラス
    """
    def __init__(self, data_path, label_path, label_map, non_mg_mapped_label, 
                 person_ids_to_load, augment=False):
        self.data_path = data_path
        self.label_path = label_path
        self.label_map = label_map
        self.non_mg_mapped_label = non_mg_mapped_label
        self.person_ids_to_load = person_ids_to_load
        self.augment = augment

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"ラベルファイルが見つかりません: {label_path}")

        self.load_data()

    def _get_person_id(self, sample_name):
        """
        サンプル名から人物IDを抽出します (例: 'Sample0010115' -> 10)
        """
        # 正規表現 'Sample(\d{4})' を使って、'Sample'の直後にある4桁の数字を抽出
        # (\d{4}) は「4桁の数字」を意味し、カッコで囲むことでその部分をグループとして取得できる
        match = re.search(r'Sample(\d{4})', sample_name)
        if match:
            # match.group(1) でキャプチャしたグループ（'0010'など）を取得し、整数に変換
            return int(match.group(1))
        
        # パターンに一致しない場合はエラーを発生させる
        raise ValueError(f"サンプル名 '{sample_name}' から人物IDを抽出できませんでした。")

    def load_data(self):
        """
        データを読み込み、指定された人物IDでフィルタリングします
        """
        with open(self.label_path, 'rb') as f:
            loaded_data = pickle.load(f)
            if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                sample_names, original_labels_all = loaded_data
            else:
                raise ValueError("ラベルファイルにはサンプル名とラベルのタプルが必要です")

        data_all = np.load(self.data_path)

        selected_indices = []
        for i, name in enumerate(sample_names):
            person_id = self._get_person_id(name)
            if person_id in self.person_ids_to_load:
                selected_indices.append(i)
        
        if not selected_indices:
            print(f"警告: 指定された人物ID {self.person_ids_to_load} に合致するデータが見つかりませんでした。")

        self.data = data_all[selected_indices]
        self.original_labels = [int(original_labels_all[i]) for i in selected_indices]
        
        self.mapped_labels = [self.label_map[orig_label] for orig_label in self.original_labels]
        self.binary_labels = [0 if mapped_label == self.non_mg_mapped_label else 1 for mapped_label in self.mapped_labels]

        self.num_samples = len(self.original_labels)
        if self.num_samples > 0:
            self.N, self.C, self.T, self.V, self.M = self.data.shape
            assert self.N == self.num_samples, "データのサンプル数とラベルのサンプル数が一致しません"

    def __len__(self):
        return self.num_samples * 2 if self.augment else self.num_samples

    def __getitem__(self, index):
        orig_index = index % self.num_samples
        flip = self.augment and (index >= self.num_samples)
        
        data_numpy = self.data[orig_index]
        if flip:
            data_numpy = horizontal_flip_skeleton(data_numpy)
            
        mapped_label = self.mapped_labels[orig_index]
        binary_label = self.binary_labels[orig_index]
        
        data_tensor = torch.from_numpy(data_numpy).float()
        
        return data_tensor, mapped_label, binary_label

class TrainSkeletonDataset(BaseSkeletonDataset):
    """
    学習用データセット (人物ID: 1-30)
    データ拡張はデフォルトで有効です。
    """
    def __init__(self, data_path, label_path, label_map, non_mg_mapped_label):
        # 学習用には1番から30番目の人物を使用
        person_ids = list(range(1, 31))
        super().__init__(
            data_path=data_path,
            label_path=label_path,
            label_map=label_map,
            non_mg_mapped_label=non_mg_mapped_label,
            person_ids_to_load=person_ids,
            augment=True  # 学習時はデータ拡張を有効にする
        )

class ValidationSkeletonDataset(BaseSkeletonDataset):
    """
    検証用データセット (人物ID: 31-35)
    データ拡張はデフォルトで無効です。
    """
    def __init__(self, data_path, label_path, label_map, non_mg_mapped_label):
        # 検証用には31番から35番目の人物を使用
        person_ids = list(range(31, 36))
        super().__init__(
            data_path=data_path,
            label_path=label_path,
            label_map=label_map,
            non_mg_mapped_label=non_mg_mapped_label,
            person_ids_to_load=person_ids,
            augment=False  # 検証時はデータ拡張を無効にする
        )


class UnifiedSkeletonDataset(Dataset):
    def __init__(self, data_path, label_path, label_map, non_mg_mapped_label, augment=True):
        self.data_path = data_path
        self.label_path = label_path
        self.augment = augment
        self.label_map = label_map
        self.non_mg_mapped_label = non_mg_mapped_label
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"ラベルファイルが見つかりません: {label_path}")
            
        self.load_data()
        self.num_samples = len(self.original_labels)

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            loaded_data = pickle.load(f)
            if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                _, self.original_labels = loaded_data
            else:
                self.original_labels = loaded_data
            self.original_labels = [int(label) for label in self.original_labels]

        # マップされたラベルとバイナリラベルを事前に作成
        self.mapped_labels = [self.label_map[orig_label] for orig_label in self.original_labels]
        self.binary_labels = [0 if mapped_label == self.non_mg_mapped_label else 1 for mapped_label in self.mapped_labels]

        self.data = np.load(self.data_path)
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        if self.N != len(self.original_labels):
            self.N = len(self.original_labels)
            self.data = self.data[:self.N]

    def __len__(self):
        return self.num_samples * 2 if self.augment else self.num_samples

    def __getitem__(self, index):
        orig_index = index % self.num_samples
        flip = self.augment and (index >= self.num_samples)
        
        data_numpy = self.data[orig_index]
        if flip:
            data_numpy = horizontal_flip_skeleton(data_numpy)
            
        mapped_label = self.mapped_labels[orig_index]
        binary_label = self.binary_labels[orig_index]
        
        data_tensor = torch.from_numpy(data_numpy).float()
        
        # マルチクラスラベルとバイナリラベルの両方を返す
        return data_tensor, mapped_label, binary_label

# 2つの損失を計算し、合計して学習するように変更
def train_epoch(model, data_loader, criterion_multiclass, criterion_binary, optimizer, device, epoch_num, num_epochs, loss_alpha):
    model.train()
    running_loss = 0.0
    running_loss_mc = 0.0 # Multi-class loss
    running_loss_bin = 0.0 # Binary loss
    correct_predictions_mc = 0
    correct_predictions_bin = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_num}/{num_epochs} Training', leave=False)
    for inputs, mapped_labels, binary_labels in progress_bar:
        inputs = inputs.to(device)
        mapped_labels = mapped_labels.to(device).long()
        binary_labels = binary_labels.to(device).long()
        
        optimizer.zero_grad()
        
        outputs = model(inputs) # Shape: (batch, num_classes) e.g., (16, 17)
        
        # 1. マルチクラス損失 (細かく分けるロス)
        loss_mc = criterion_multiclass(outputs, mapped_labels)
        
        # 2. バイナリ損失 (大雑把に分けるロス)
        # non-MGのロジットと、全MGクラスのロジットの合計(logsumexp)で新しい2次元のロジットを作成
        non_mg_mapped_label = data_loader.dataset.dataset.non_mg_mapped_label if isinstance(data_loader.dataset, Subset) else data_loader.dataset.non_mg_mapped_label
        
        mg_indices = [i for i in range(outputs.shape[1]) if i != non_mg_mapped_label]
        
        non_mg_logit = outputs[:, non_mg_mapped_label]
        mg_logits_sum = torch.logsumexp(outputs[:, mg_indices], dim=1)
        binary_logits = torch.stack([non_mg_logit, mg_logits_sum], dim=1)
        
        loss_bin = criterion_binary(binary_logits, binary_labels)
        
        # 3. 合計損失
        total_loss = loss_mc + loss_alpha * loss_bin
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item() * inputs.size(0)
        running_loss_mc += loss_mc.item() * inputs.size(0)
        running_loss_bin += loss_bin.item() * inputs.size(0)
        
        # 精度計算
        _, predicted_mc = torch.max(outputs.data, 1)
        _, predicted_bin = torch.max(binary_logits.data, 1)
        
        total_samples += mapped_labels.size(0)
        correct_predictions_mc += (predicted_mc == mapped_labels).sum().item()
        correct_predictions_bin += (predicted_bin == binary_labels).sum().item()
        
        acc_mc = correct_predictions_mc / total_samples
        acc_bin = correct_predictions_bin / total_samples
        
        progress_bar.set_postfix(
            loss=f'{total_loss.item():.4f}',
            acc_mc=f'{acc_mc:.4f}',
            acc_bin=f'{acc_bin:.4f}'
        )
        
    epoch_loss = running_loss / total_samples
    epoch_loss_mc = running_loss_mc / total_samples
    epoch_loss_bin = running_loss_bin / total_samples
    epoch_acc_mc = correct_predictions_mc / total_samples
    epoch_acc_bin = correct_predictions_bin / total_samples
    
    return epoch_loss, epoch_loss_mc, epoch_loss_bin, epoch_acc_mc, epoch_acc_bin

def evaluate_epoch(model, data_loader, criterion_multiclass, criterion_binary, device, loss_alpha):
    model.eval()
    running_loss = 0.0
    correct_predictions_mc = 0
    total_samples = 0
    progress_bar = tqdm(data_loader, desc='Validation', leave=False)
    with torch.no_grad():
        for inputs, mapped_labels, binary_labels in progress_bar:
            inputs = inputs.to(device)
            mapped_labels = mapped_labels.to(device).long()
            outputs = model(inputs)
            loss_mc = criterion_multiclass(outputs, mapped_labels)
            running_loss += loss_mc.item() * inputs.size(0)
            _, predicted_mc = torch.max(outputs.data, 1)
            total_samples += mapped_labels.size(0)
            correct_predictions_mc += (predicted_mc == mapped_labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions_mc / total_samples
    return epoch_loss, epoch_acc

def train_unified_model(model, train_loader, val_loader, criterion_multiclass, criterion_binary, optimizer, scheduler, device, num_epochs, save_dir, loss_alpha, save_interval=10):
    print("\n--- Unified Multi-Task Training Started ---")
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_loss_mc': [],  # 追加
        'train_loss_bin': []  # 追加
    }
    
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        train_loss, train_loss_mc, train_loss_bin, train_acc_mc, train_acc_bin = train_epoch(
            model, train_loader, criterion_multiclass, criterion_binary, optimizer, device, epoch + 1, num_epochs, loss_alpha
        )
        print(f"Epoch {epoch+1} Train Summary: Total Loss={train_loss:.4f}, MC Acc={train_acc_mc:.4f}, Bin Acc={train_acc_bin:.4f}")
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc_mc)
        history['train_loss_mc'].append(train_loss_mc)   
        history['train_loss_bin'].append(train_loss_bin) 

        val_loss, val_acc = evaluate_epoch(
            model, val_loader, criterion_multiclass, criterion_binary, device, loss_alpha
        )
        print(f"Epoch {epoch+1} Validation Summary: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1} Time: {time.time() - epoch_start_time:.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, 'best_unified_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"*** Best validation accuracy updated: {best_val_acc:.4f}. Model saved to {best_model_path} ***")

        if (epoch + 1) % save_interval == 0:
            save_path_epoch = os.path.join(save_dir, f'epoch_{epoch+1}_unified_model.pth')
            torch.save(model.state_dict(), save_path_epoch)
            print(f"Model saved to {save_path_epoch}")

    print("\n--- Unified Multi-Task Training Finished ---")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    # 損失と精度のプロット
    plot_training_history(history['train_loss'], history['val_loss'], "Loss", os.path.join(save_dir, 'unified_loss_history.png'))
    plot_training_history(history['train_acc'], history['val_acc'], "Accuracy", os.path.join(save_dir, 'unified_accuracy_history.png'))

    plot_training_history(
        history['train_loss_mc'], None, "Multi-class Loss", 
        os.path.join(save_dir, 'unified_multiclass_loss_history.png'), stage_name="Unified"
    )
    plot_training_history(
        history['train_loss_bin'], None, "Binary Loss", 
        os.path.join(save_dir, 'unified_binary_loss_history.png'), stage_name="Unified"
    )

def test_unified_model(model, data_loader, device, mapped_to_original_map, output_file_path):
    model.eval()
    
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    all_true_labels = []
    all_pred_labels_top1 = []

    # ★★★ カテゴリごとの正解数と総数を記録するリストを初期化 ★★★
    # クラス数を取得 (モデルの出力層のサイズから)
    # data_loaderから最初のバッチを取得して形状を確認する
    try:
        sample_inputs, _, _ = next(iter(data_loader))
        num_classes = model(sample_inputs.to(device)).shape[1]
    except StopIteration:
        print("データローダーが空です。テストをスキップします。")
        return
        
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    # ★★★ ここまで ★★★
    
    progress_bar = tqdm(data_loader, desc='Testing Unified Model', leave=False)
    with torch.no_grad():
        for inputs, mapped_labels, _ in progress_bar: 
            inputs = inputs.to(device)
            mapped_labels = mapped_labels.to(device) 
            
            # --- フォワードパス ---
            outputs = model(inputs) # Shape: (batch, num_classes)
            
            # --- Top-1 と Top-5 の精度計算 ---
            _, top5_preds = torch.topk(outputs, 5, dim=1)
            mapped_labels_reshaped = mapped_labels.view(-1, 1)
            correct_top5_batch = (top5_preds == mapped_labels_reshaped).any(dim=1)
            correct_top5 += correct_top5_batch.sum().item()

            top1_preds = top5_preds[:, 0]
            correct_top1 += (top1_preds == mapped_labels).sum().item()

            total_samples += mapped_labels.size(0)
            
            all_true_labels.extend(mapped_labels.cpu().numpy())
            all_pred_labels_top1.extend(top1_preds.cpu().numpy())

            # ★★★ カテゴリごとの正解数と総数を更新 ★★★
            # バッチ内の各サンプルのTop-1予測が正解かどうかをブール値で取得
            correct_mask = (top1_preds == mapped_labels).squeeze()
            
            # バッチ内の各サンプルについてループ
            for i in range(len(mapped_labels)):
                label = mapped_labels[i].item() # 正解ラベル
                class_total[label] += 1
                print(correct_mask[i])
                pri
                if correct_mask[i].item(): # 予測が正しければ
                    class_correct[label] += 1
            # ★★★ ここまで ★★★

            progress_bar.set_postfix(
                top1=f'{correct_top1 / total_samples:.4f}',
                top5=f'{correct_top5 / total_samples:.4f}'
            )

    # --- 最終的な精度の計算と表示 ---
    top1_accuracy = correct_top1 / total_samples
    top5_accuracy = correct_top5 / total_samples
    
    print("\n--- Overall Test Results ---")
    print(f"Total Samples: {total_samples}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
    print("----------------------------")

    # ★★★ カテゴリごとの正解率を表示 ★★★
    print("\n--- Per-Category Accuracy (Recall) ---")
    for i in range(num_classes):
        # mapped_to_original_map を使って元のラベル名を取得
        original_label = mapped_to_original_map.get(i, f"Unknown({i})")
        
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f"Class {i} (Original: {original_label}): {accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            # テストセットにそのカテゴリのサンプルがなかった場合
            print(f"Class {i} (Original: {original_label}): N/A (0 samples)")
    print("----------------------------------------")

    # --- 混同行列の作成 ---
    # マップされたラベルを元のラベルに戻す
    all_true_original = [mapped_to_original_map[l] for l in all_true_labels]
    all_pred_original_top1 = [mapped_to_original_map[l] for l in all_pred_labels_top1]
    
    unique_labels = sorted(list(mapped_to_original_map.values()))
    cm = confusion_matrix(all_true_original, all_pred_original_top1, labels=unique_labels)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("--- Test Results ---\n")
            
            f.write(f"Total Samples: {total_samples}\n")
            f.write(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)\n")
            f.write(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)\n\n")
            
            f.write("--- Confusion Matrix (Original Labels) ---\n")
            # ヘッダー行 (Predicted Labels)
            header = "True\\Pred | " + " ".join([f"{label:^4}" for label in unique_labels]) + "\n"
            f.write(header)
            f.write("-" * len(header) + "\n")
            
            # 各行 (True Label and counts)
            for i, true_label in enumerate(unique_labels):
                row_str = f"{true_label:^9} | " + " ".join([f"{count:^4}" for count in cm[i]]) + "\n"
                f.write(row_str)
            
        print(f"Test results saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving results to file: {e}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted Label (Original)')
    plt.ylabel('True Label (Original)')
    plt.title('Unified Model Confusion Matrix (Top-1 Predictions)')
    plt.savefig('unified_model_confusion_matrix.png')
    print("Confusion matrix saved to unified_model_confusion_matrix.png")
    plt.close()

def plot_training_history(train_metric, val_metric, metric_name, save_path, stage_name="Unified"):
    epochs = range(1, len(train_metric) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metric, 'bo-', label=f'Training {metric_name}')
    if val_metric is not None:
        plt.plot(epochs, val_metric, 'ro-', label=f'Validation {metric_name}')
    plt.title(f'{stage_name} Training and Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"{metric_name} plot saved to {save_path}")

if __name__ == '__main__':
    # --- データパス ---
    data_path = 'SMGskeleton/train_data.npy'
    label_path = 'SMGskeleton/train_label.pkl'
    test_data_path = 'SMGskeleton/test_data.npy'
    test_label_path = 'SMGskeleton/test_label.pkl'
    
    ORIGINAL_NON_MG_LABEL = 8 # 元のデータでのnon-MGラベル
    
    # --- ハイパーパラメータ ---
    batch_size = 16
    num_workers = 0
    learning_rate = 0.01
    num_epochs = 30
    save_interval = 10
    model_save_dir = './msg3d_unified_model' 
    loss_alpha = 0.5# バイナリ損失の重み
    run_training =False
    run_test = True

    # MSG3D Model parameters
    num_gcn_scales = 13
    num_g3d_scales = 6
    from graph.ntu_rgb_d_msg3d import AdjMatrixGraph # Ensure this import path is correct

    graph_type = 'graph.ntu_rgb_d_msg3d.AdjMatrixGraph'

    os.makedirs(model_save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 使用デバイス: {device} ---")

    # --- ラベルマッピングの作成 ---
    # non-MGラベルを最後に、MGラベルを0から連続になるようにマッピング
    print("\n--- ラベルマッピングの作成 ---")
    with open(label_path, 'rb') as f:
        _, original_labels_list = pickle.load(f)
    
    unique_original_labels = sorted(list(set(original_labels_list)))
    mg_labels = [l for l in unique_original_labels if l != ORIGINAL_NON_MG_LABEL]
    
    num_classes = len(unique_original_labels)
    NON_MG_MAPPED_LABEL = num_classes - 1 # non-MGは最後のインデックスに
    
    original_to_mapped_map = {label: i for i, label in enumerate(mg_labels)}
    original_to_mapped_map[ORIGINAL_NON_MG_LABEL] = NON_MG_MAPPED_LABEL
    mapped_to_original_map = {v: k for k, v in original_to_mapped_map.items()}

    print(f"全クラス数: {num_classes}")
    print(f"ラベルマップ (Original -> Mapped): {original_to_mapped_map}")
    print(f"Non-MG ラベル '{ORIGINAL_NON_MG_LABEL}' は '{NON_MG_MAPPED_LABEL}' にマップされました。")

    # --- データローダーの準備 ---
    print("\n--- データローダーの準備 ---")
    # 学習用データセット
    train_dataset = TrainSkeletonDataset(
        data_path=data_path,
        label_path=label_path,
        label_map=original_to_mapped_map,
        non_mg_mapped_label=NON_MG_MAPPED_LABEL
    )

    # 検証用データセット
    val_dataset = ValidationSkeletonDataset(
        data_path=data_path,
        label_path= label_path,
        label_map=original_to_mapped_map,
        non_mg_mapped_label=NON_MG_MAPPED_LABEL
    )

    print(f"学習用サンプル数: {len(train_dataset)}")
    print(f"検証用サンプル数: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train Samples: {len(train_dataset)}, Val Samples: {len(val_dataset)}")

    # --- モデル、損失、オプティマイザの初期化 ---
    print("\n--- モデル、損失関数、オプティマイザの初期化 ---")
    in_channels = train_dataset.C
    num_point = train_dataset.V
    num_person = train_dataset.M

    model = MSG3D_Model(
        num_class=num_classes, # 全クラス数
        num_point=num_point, num_person=num_person,
        num_gcn_scales=num_gcn_scales, num_g3d_scales=num_g3d_scales,
        graph=graph_type,
        in_channels=in_channels
    ).to(device)

    #criterion_multiclass = nn.CrossEntropyLoss()
    #criterion_binary = nn.CrossEntropyLoss()
    criterion_multiclass = FocalLoss(gamma=2.0, alpha=None, reduction='mean')
    criterion_binary = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0001)
        # --- モデル、損失、オプティマイザの初期化 ---
    print("\n--- モデル、損失関数、オプティマイザの初期化 ---")
    # ... (model, criterion, optimizer の定義はそのまま) ...
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0001)
    
    # 'val_loss'が'patience'エポック数改善しなかったら学習率を0.1倍（factor）にする
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      # 'val_loss'を監視するので'min'。精度を監視するなら'max'
        factor=0.1,      # 学習率を0.1倍に下げる
        patience=5,      # 5エポック改善が見られなければ実行
    )

    # --- 学習フェーズ ---
    if run_training:
        train_unified_model(
            model, train_loader, val_loader,
            criterion_multiclass, criterion_binary,
            optimizer, scheduler, device, num_epochs,
            model_save_dir, loss_alpha, save_interval
        )
    else:
        print("\n--- 学習はスキップされました ---")

    # --- テストフェーズ ---
    if run_test:
        print("\n--- テスト開始 ---")
        test_dataset = UnifiedSkeletonDataset(
            data_path=test_data_path, label_path=test_label_path,
            label_map=original_to_mapped_map, non_mg_mapped_label=NON_MG_MAPPED_LABEL, augment=False
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        model_path = os.path.join(model_save_dir, 'best_unified_model.pth')
        #model_path = os.path.join(model_save_dir, 'epoch_20_unified_model.pth')

        print(f"テストモデルをロード: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_unified_model(model, test_loader, device, mapped_to_original_map,output_file_path='test_results.txt')
    else:
        print("\n--- テストはスキップされました ---")