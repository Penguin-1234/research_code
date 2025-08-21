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
# from net.utils.graph import Graph # graph.ntu_rgb_d_msg3d を直接指定するため不要
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import re

class EnsembleMSG3D(nn.Module):
    """
    複数のデータストリーム（joint, boneなど）を受け取り、
    それぞれを個別のMSG3Dモデルで処理し、出力をアンサンブルするモデル。
    """
    def __init__(self, use_streams, num_class, num_point, num_person,
                 num_gcn_scales, num_g3d_scales, graph, in_channels):
        super().__init__()

        self.use_streams = {k: v for k, v in use_streams.items() if v}
        if not self.use_streams:
            raise ValueError("少なくとも1つのデータストリームを有効にする必要があります。")

        self.models = nn.ModuleDict()
        print("\n--- アンサンブルモデルの初期化 ---")
        for stream_name in self.use_streams.keys():
            print(f"  - ストリーム '{stream_name}' のモデルを初期化中...")
            self.models[stream_name] = MSG3D_Model(
                num_class=num_class,
                num_point=num_point,
                num_person=num_person,
                num_gcn_scales=num_gcn_scales,
                num_g3d_scales=num_g3d_scales,
                graph=graph,
                in_channels=in_channels
            )
        print("---------------------------------")

    def forward(self, x_dict):
        """
        x_dict: データストリーム名をキー、テンソルを値とする辞書
                例: {'joint': tensor, 'bone': tensor}
        """
        outputs = []
        for stream_name, model in self.models.items():
            if stream_name not in x_dict:
                raise ValueError(f"モデルはストリーム '{stream_name}' を予期していますが、入力データに含まれていません。")

            stream_output = model(x_dict[stream_name])
            outputs.append(stream_output)

        ensembled_output = torch.mean(torch.stack(outputs), dim=0)
        return ensembled_output

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

def horizontal_flip_skeleton(data_numpy):
    left_right_pairs = [
        (4, 8), (5, 7), (6, 10), (11, 17), (12, 16), (13, 15), (23, 21), (24, 22)
    ]
    data_flipped = np.copy(data_numpy)
    data_flipped[0, :, :, :] *= -1
    for left, right in left_right_pairs:
        data_flipped[:, :, [left, right], :] = data_flipped[:, :, [right, left], :]
    return data_flipped

def temporal_reverse_skeleton(data_numpy):
    return data_numpy[:, ::-1, :, :].copy()

class BaseSkeletonDataset(Dataset):
    def __init__(self, data_path, label_path, label_map, category_map, non_mg_mapped_label,
                 person_ids_to_load, use_bone=False, use_delta_joint=False, use_delta_bone=False, augment=False):
        self.data_path = data_path
        self.label_path = label_path
        self.label_map = label_map
        self.category_map = category_map
        self.non_mg_mapped_label = non_mg_mapped_label
        self.person_ids_to_load = person_ids_to_load
        self.use_bone = use_bone
        self.use_delta_joint = use_delta_joint
        self.use_delta_bone = use_delta_bone
        self.augment = augment

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"ラベルファイルが見つかりません: {label_path}")

        self.load_data()

    def _get_person_id(self, sample_name):
        match = re.search(r'Sample(\d{4})', sample_name)
        if match:
            return int(match.group(1))
        raise ValueError(f"サンプル名 '{sample_name}' から人物IDを抽出できませんでした。")

    def compute_bone(self, data_tensor):
        bone_pairs = [
            (1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7),
            (9, 0), (10, 9), (11, 10), (12, 11), (13, 0), (14, 13), (15, 14), (16, 15),
            (17, 14), (18, 15), (19, 16), (20, 0), (21, 20), (22, 21), (23, 22), (24, 23)
        ]
        C, T, V, M = data_tensor.shape
        bone = torch.zeros_like(data_tensor)
        for child, parent in bone_pairs:
            bone[:, :, child] = data_tensor[:, :, child] - data_tensor[:, :, parent]
        return bone

    def temporal_diff(self, x):
        delta = x[:, 1:] - x[:, :-1]
        pad = torch.zeros_like(delta[:, :1])
        return torch.cat([delta, pad], dim=1)

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            loaded_data = pickle.load(f)
            if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                sample_names, original_labels_all = loaded_data
            else:
                raise ValueError("ラベルファイルにはサンプル名とラベルのタプルが必要です")

        data_all = np.load(self.data_path)
        selected_indices = [i for i, name in enumerate(sample_names) if self._get_person_id(name) in self.person_ids_to_load]
        if not selected_indices:
            print(f"警告: 指定された人物ID {self.person_ids_to_load} に合致するデータが見つかりませんでした。")

        self.data = data_all[selected_indices]
        self.original_labels = [int(original_labels_all[i]) for i in selected_indices]
        self.mapped_labels = [self.label_map[orig_label] for orig_label in self.original_labels]
        self.category_labels = [self.category_map[orig_label] for orig_label in self.original_labels]
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
        data_tensor = torch.from_numpy(data_numpy).float()
        output = {"joint": data_tensor}
        if self.use_bone or self.use_delta_bone:
            bone = self.compute_bone(data_tensor)
            if self.use_bone:
                output["bone"] = bone
            if self.use_delta_bone:
                output["delta_bone"] = self.temporal_diff(bone)
        if self.use_delta_joint:
            output["delta_joint"] = self.temporal_diff(data_tensor)
        mapped_label = self.mapped_labels[orig_index]
        category_label = self.category_labels[orig_index]
        # 返り値を (入力辞書, マルチクラスラベル, カテゴリラベル) に統一
        return output, mapped_label, category_label

class TrainSkeletonDataset(BaseSkeletonDataset):
    def __init__(self, data_path, label_path, label_map, category_map, non_mg_mapped_label,use_bone,use_delta_joint,use_delta_bone):
        super().__init__(data_path, label_path, label_map, category_map, non_mg_mapped_label, list(range(1, 31)), use_bone, use_delta_joint, use_delta_bone, True)

class ValidationSkeletonDataset(BaseSkeletonDataset):
    def __init__(self, data_path, label_path, label_map, category_map, non_mg_mapped_label, use_bone, use_delta_joint, use_delta_bone):
        super().__init__(data_path, label_path, label_map, category_map, non_mg_mapped_label, list(range(31, 36)), use_bone, use_delta_joint, use_delta_bone, False)

class UnifiedSkeletonDataset(Dataset):
    def __init__(self, data_path, label_path, label_map, category_map, non_mg_mapped_label,
                 use_bone, use_delta_joint, use_delta_bone, augment=False):
        self.data_path = data_path
        self.label_path = label_path
        self.augment = augment
        self.label_map = label_map
        self.category_map = category_map
        self.use_bone = use_bone
        self.use_delta_joint = use_delta_joint
        self.use_delta_bone = use_delta_bone
        if not os.path.exists(data_path): raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")
        if not os.path.exists(label_path): raise FileNotFoundError(f"ラベルファイルが見つかりません: {label_path}")
        self.load_data()

    def compute_bone(self, data_tensor):
        bone_pairs = [(1,0),(2,1),(3,2),(4,3),(5,0),(6,5),(7,6),(8,7),(9,0),(10,9),(11,10),(12,11),(13,0),(14,13),(15,14),(16,15),(17,14),(18,15),(19,16),(20,0),(21,20),(22,21),(23,22),(24,23)]
        C, T, V, M = data_tensor.shape
        bone = torch.zeros_like(data_tensor)
        for child, parent in bone_pairs: bone[:, :, child] = data_tensor[:, :, child] - data_tensor[:, :, parent]
        return bone

    def temporal_diff(self, x):
        delta = x[:, 1:] - x[:, :-1]; pad = torch.zeros_like(delta[:, :1]); return torch.cat([delta, pad], dim=1)

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            _, self.original_labels = pickle.load(f)
        self.original_labels = [int(label) for label in self.original_labels]
        self.mapped_labels = [self.label_map[ol] for ol in self.original_labels]
        self.category_labels = [self.category_map[ol] for ol in self.original_labels]
        self.data = np.load(self.data_path)
        self.num_samples = len(self.original_labels)
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        if self.N != self.num_samples: print(f"警告: データ数({self.N})とラベル数({self.num_samples})が異なります。"); self.N = self.num_samples; self.data = self.data[:self.N]

    def __len__(self):
        return self.num_samples * 2 if self.augment else self.num_samples

    def __getitem__(self, index):
        orig_index = index % self.num_samples
        flip = self.augment and (index >= self.num_samples)
        data_numpy = self.data[orig_index]
        if flip: data_numpy = horizontal_flip_skeleton(data_numpy)
        data_tensor = torch.from_numpy(data_numpy).float()
        output = {"joint": data_tensor}
        if self.use_bone or self.use_delta_bone:
            bone = self.compute_bone(data_tensor)
            if self.use_bone: output["bone"] = bone
            if self.use_delta_bone: output["delta_bone"] = self.temporal_diff(bone)
        if self.use_delta_joint: output["delta_joint"] = self.temporal_diff(data_tensor)
        mapped_label = self.mapped_labels[orig_index]
        category_label = self.category_labels[orig_index]
        return output, mapped_label, category_label

def collate_fn(batch):
    data_dicts, mapped_labels, category_labels = zip(*batch)
    batch_data = {}
    if data_dicts:
        for key in data_dicts[0].keys():
            batch_data[key] = torch.stack([d[key] for d in data_dicts])
    return batch_data, torch.tensor(mapped_labels, dtype=torch.long), torch.tensor(category_labels, dtype=torch.long)

def train_epoch(model, data_loader, criterion_multiclass, criterion_category, optimizer, device,
                epoch_num, num_epochs, loss_alpha, category_indices):
    model.train()
    running_loss, running_loss_mc, running_loss_category = 0.0, 0.0, 0.0
    correct_predictions_mc, correct_predictions_category, total_samples = 0, 0, 0

    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_num}/{num_epochs} Training', leave=False)
    for inputs_dict, mapped_labels, category_labels in progress_bar:
        inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
        mapped_labels = mapped_labels.to(device)
        category_labels = category_labels.to(device)
        batch_size = mapped_labels.size(0)

        optimizer.zero_grad()
        outputs = model(inputs_dict)

        # 1. Multi-class classification loss (クラスごとの損失)
        loss_mc = criterion_multiclass(outputs, mapped_labels)

        # 2. Category classification loss (カテゴリごとの損失)
        category_logits_list = []
        for cat_id in sorted(category_indices.keys()):
            indices = category_indices[cat_id]
            if len(indices) > 0:
                cat_logit = torch.logsumexp(outputs[:, indices], dim=1)
                category_logits_list.append(cat_logit)
            else:
                category_logits_list.append(torch.full_like(outputs[:, 0], -1e9))

        category_logits = torch.stack(category_logits_list, dim=1)
        loss_category = criterion_category(category_logits, category_labels)

        # 2つの損失を重み付けして合計
        total_loss = loss_mc + loss_alpha * loss_category
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item() * batch_size
        running_loss_mc += loss_mc.item() * batch_size
        running_loss_category += loss_category.item() * batch_size

        _, predicted_mc = torch.max(outputs.data, 1)
        _, predicted_category = torch.max(category_logits.data, 1)

        total_samples += batch_size
        correct_predictions_mc += (predicted_mc == mapped_labels).sum().item()
        correct_predictions_category += (predicted_category == category_labels).sum().item()

        progress_bar.set_postfix(
            loss=f'{total_loss.item():.4f}',
            acc_mc=f'{correct_predictions_mc / total_samples:.4f}',
            acc_cat=f'{correct_predictions_category / total_samples:.4f}'
        )

    return (running_loss / total_samples, running_loss_mc / total_samples, running_loss_category / total_samples,
            correct_predictions_mc / total_samples, correct_predictions_category / total_samples)

def evaluate_epoch(model, data_loader, criterion_multiclass, device):
    model.eval()
    running_loss, correct_predictions_mc, total_samples = 0.0, 0, 0
    progress_bar = tqdm(data_loader, desc='Validation', leave=False)
    with torch.no_grad():
        for inputs_dict, mapped_labels, _ in progress_bar: # category_labelsは評価では使用しない
            inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
            mapped_labels = mapped_labels.to(device)
            batch_size = mapped_labels.size(0)

            outputs = model(inputs_dict)
            loss_mc = criterion_multiclass(outputs, mapped_labels)

            running_loss += loss_mc.item() * batch_size
            _, predicted_mc = torch.max(outputs.data, 1)
            total_samples += batch_size
            correct_predictions_mc += (predicted_mc == mapped_labels).sum().item()

    return running_loss / total_samples, correct_predictions_mc / total_samples

def train_unified_model(model, train_loader, val_loader, criterion_multiclass, criterion_category, optimizer,
                        scheduler, device, num_epochs, save_dir, loss_alpha, save_interval, category_indices):
    print("\n--- Unified Multi-Task Training Started ---")
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc_mc': [], 'val_acc_mc': [],
               'train_loss_mc': [], 'train_loss_category': [], 'train_acc_category': []}
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        train_loss, train_loss_mc, train_loss_category, train_acc_mc, train_acc_category = train_epoch(
            model, train_loader, criterion_multiclass, criterion_category, optimizer, device,
            epoch + 1, num_epochs, loss_alpha, category_indices
        )
        print(f"Epoch {epoch+1} Train Summary: Total Loss={train_loss:.4f}, MC Acc={train_acc_mc:.4f}, Cat Acc={train_acc_category:.4f}")
        history['train_loss'].append(train_loss)
        history['train_acc_mc'].append(train_acc_mc)
        history['train_loss_mc'].append(train_loss_mc)
        history['train_loss_category'].append(train_loss_category)
        history['train_acc_category'].append(train_acc_category)

        val_loss, val_acc_mc = evaluate_epoch(model, val_loader, criterion_multiclass, device)
        print(f"Epoch {epoch+1} Validation Summary: Loss={val_loss:.4f}, MC Acc={val_acc_mc:.4f}")
        history['val_loss'].append(val_loss)
        history['val_acc_mc'].append(val_acc_mc)

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1} Time: {time.time() - epoch_start_time:.2f}s")

        if val_acc_mc > best_val_acc:
            best_val_acc = val_acc_mc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_unified_model.pth'))
            print(f"*** Best validation accuracy updated: {best_val_acc:.4f}. Model saved. ***")

        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch+1}_unified_model.pth'))
            print(f"Model saved at epoch {epoch+1}")

    print("\n--- Unified Multi-Task Training Finished ---")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    plot_training_history(history['train_loss'], history['val_loss'], "Total Loss", os.path.join(save_dir, 'unified_total_loss_history.png'))
    plot_training_history(history['train_acc_mc'], history['val_acc_mc'], "Multi-Class Accuracy", os.path.join(save_dir, 'unified_mc_accuracy_history.png'))
    plot_training_history(history['train_acc_category'], None, "Category Accuracy", os.path.join(save_dir, 'unified_category_accuracy_history.png'), "Training")
    plot_training_history(history['train_loss_mc'], None, "Multi-class Loss", os.path.join(save_dir, 'unified_multiclass_loss_history.png'), "Training")
    plot_training_history(history['train_loss_category'], None, "Category Loss", os.path.join(save_dir, 'unified_category_loss_history.png'), "Training")


def test_unified_model(model, data_loader, device, mapped_to_original_map, output_file_path):
    model.eval()
    all_true_labels, all_pred_labels_top1 = [], []
    try:
        sample_inputs_dict, _, _ = next(iter(data_loader))
        sample_inputs_dict = {k: v.to(device) for k, v in sample_inputs_dict.items()}
        num_classes = model(sample_inputs_dict).shape[1]
    except StopIteration:
        print("データローダーが空です。テストをスキップします。"); return

    class_correct, class_total = list(0. for _ in range(num_classes)), list(0. for _ in range(num_classes))
    correct_top1, correct_top5, total_samples = 0, 0, 0

    progress_bar = tqdm(data_loader, desc='Testing Unified Model', leave=False)
    with torch.no_grad():
        for inputs_dict, mapped_labels, _ in progress_bar:
            inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
            mapped_labels = mapped_labels.to(device)
            batch_size = mapped_labels.size(0)

            outputs = model(inputs_dict)
            _, top5_preds = torch.topk(outputs, 5, dim=1)
            correct_top5 += (top5_preds == mapped_labels.view(-1, 1)).any(dim=1).sum().item()

            top1_preds = top5_preds[:, 0]
            correct_top1 += (top1_preds == mapped_labels).sum().item()
            total_samples += batch_size

            all_true_labels.extend(mapped_labels.cpu().numpy())
            all_pred_labels_top1.extend(top1_preds.cpu().numpy())

            correct_mask = (top1_preds == mapped_labels)
            for i in range(batch_size):
                label = mapped_labels[i].item()
                class_total[label] += 1
                if correct_mask[i].item(): class_correct[label] += 1

            progress_bar.set_postfix(top1=f'{correct_top1/total_samples:.4f}', top5=f'{correct_top5/total_samples:.4f}')

    top1_accuracy, top5_accuracy = correct_top1 / total_samples, correct_top5 / total_samples
    per_class_accuracy = [c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    macro_avg_accuracy = sum(per_class_accuracy) / len([t for t in class_total if t > 0])

    print("\n--- Overall Test Results ---")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
    print(f"Macro Average Accuracy: {macro_avg_accuracy:.4f} ({macro_avg_accuracy*100:.2f}%)")

    print("\n--- Per-Category Accuracy (Recall) ---")
    for i in range(num_classes):
        original_label = mapped_to_original_map.get(i, f"Unknown({i})")
        if class_total[i] > 0: print(f"Class {i} (Orig: {original_label}): {100*per_class_accuracy[i]:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else: print(f"Class {i} (Orig: {original_label}): N/A (0 samples)")

    all_true_original = [mapped_to_original_map[l] for l in all_true_labels]
    all_pred_original_top1 = [mapped_to_original_map[l] for l in all_pred_labels_top1]
    unique_labels = sorted(list(mapped_to_original_map.values()))
    cm = confusion_matrix(all_true_original, all_pred_original_top1, labels=unique_labels)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("--- Test Results ---\n")
        f.write(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)\n")
        f.write(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)\n")
        f.write(f"Macro Average Accuracy: {macro_avg_accuracy:.4f} ({macro_avg_accuracy*100:.2f}%)\n\n")
        f.write("--- Confusion Matrix (Original Labels) ---\n")
        header = "T\\P | " + " ".join([f"{l:^4}" for l in unique_labels]) + "\n"; f.write(header + "-"*len(header) + "\n")
        for i, true_label in enumerate(unique_labels): f.write(f"{true_label:^3} | " + " ".join([f"{c:^4}" for c in cm[i]]) + "\n")
    print(f"Test results saved to: {output_file_path}")

    plt.figure(figsize=(12, 10)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted Label (Original)'); plt.ylabel('True Label (Original)'); plt.title('Confusion Matrix')
    plt.savefig('unified_model_confusion_matrix.png'); plt.close()
    print("Confusion matrix saved to unified_model_confusion_matrix.png")

def plot_training_history(train_metric, val_metric, metric_name, save_path, stage_name=""):
    plt.figure(figsize=(10, 6)); plt.plot(range(1, len(train_metric) + 1), train_metric, 'bo-', label=f'Training {metric_name}')
    if val_metric: plt.plot(range(1, len(val_metric) + 1), val_metric, 'ro-', label=f'Validation {metric_name}')
    plt.title(f'{stage_name} {metric_name} History'); plt.xlabel('Epochs'); plt.ylabel(metric_name); plt.legend(); plt.grid(True)
    plt.savefig(save_path); plt.close()
    print(f"{metric_name} plot saved to {save_path}")

if __name__ == '__main__':
    data_path = 'SMGskeleton/train_data.npy'
    label_path = 'SMGskeleton/train_label.pkl'
    test_data_path = 'SMGskeleton/test_data.npy'
    test_label_path = 'SMGskeleton/test_label.pkl'
    ORIGINAL_NON_MG_LABEL = 8

    batch_size = 64
    num_workers = 0
    learning_rate = 0.01
    num_epochs = 30
    save_interval = 5
    model_save_dir = './msg3d_group_model'
    loss_alpha = 0.5
    run_training = True
    run_test = True

    use_joint = True  
    use_bone = True
    use_delta_joint = True
    use_delta_bone = True

    use_streams = {'joint': True, 'bone': True, 'delta_joint': True, 'delta_bone': True}
    num_gcn_scales, num_g3d_scales = 13, 6
    from graph.ntu_rgb_d_msg3d import AdjMatrixGraph # Ensure this import path is correct
    graph_type = 'graph.ntu_rgb_d_msg3d.AdjMatrixGraph'

    os.makedirs(model_save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 使用デバイス: {device} ---")

    with open(label_path, 'rb') as f: _, original_labels_list = pickle.load(f)
    unique_original_labels = sorted(list(set(original_labels_list)))
    mg_labels = [l for l in unique_original_labels if l != ORIGINAL_NON_MG_LABEL]
    num_classes = len(unique_original_labels)
    NON_MG_MAPPED_LABEL = num_classes - 1
    original_to_mapped_map = {label: i for i, label in enumerate(mg_labels)}
    original_to_mapped_map[ORIGINAL_NON_MG_LABEL] = NON_MG_MAPPED_LABEL
    mapped_to_original_map = {v: k for k, v in original_to_mapped_map.items()}

    category_map = {
        ORIGINAL_NON_MG_LABEL: 0, 0: 1, 4: 1, 9: 1, 3: 2, 5: 2, 6: 2, 10: 2, 14: 2, 16: 2,
        1: 3, 11: 3, 12: 3, 2: 4, 7: 4, 13: 4, 15: 4
    }

    num_categories = len(set(category_map.values()))
    print(f"全クラス数: {num_classes}, 全カテゴリ数: {num_categories}")
    print(f"ラベルマップ (Original -> Mapped): {original_to_mapped_map}")

    mapped_to_category_map = {m: category_map[o] for o, m in original_to_mapped_map.items()}
    category_indices = {i: [m for m, c in mapped_to_category_map.items() if c == i] for i in range(num_categories)}
    category_indices_tensors = {cat: torch.tensor(indices, dtype=torch.long, device=device) for cat, indices in category_indices.items()}
    print("カテゴリごとのインデックス (Mapped Labels):", category_indices)

    print("\n--- データローダーの準備 ---")
    train_dataset = TrainSkeletonDataset(data_path, label_path, original_to_mapped_map, category_map, NON_MG_MAPPED_LABEL, use_bone=use_bone,
        use_delta_joint=use_delta_joint, use_delta_bone=use_delta_bone)
    val_dataset = ValidationSkeletonDataset(data_path, label_path, original_to_mapped_map, category_map, NON_MG_MAPPED_LABEL, use_bone=use_bone,
        use_delta_joint=use_delta_joint, use_delta_bone=use_delta_bone)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    print(f"Train Samples: {len(train_dataset)}, Val Samples: {len(val_dataset)}")

    print("\n--- モデル、損失関数、オプティマイザの初期化 ---")
    in_channels, num_point, num_person = train_dataset.C, train_dataset.V, train_dataset.M
    model = EnsembleMSG3D(use_streams, num_classes, num_point, num_person, num_gcn_scales, num_g3d_scales, graph_type, in_channels).to(device)

    criterion_multiclass = FocalLoss(gamma=2.0)
    criterion_category = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    if run_training:
        train_unified_model(
            model, train_loader, val_loader, criterion_multiclass, criterion_category,
            optimizer, scheduler, device, num_epochs, model_save_dir, loss_alpha,
            save_interval, category_indices_tensors
        )
    else:
        print("\n--- 学習はスキップされました ---")

    if run_test:
        print("\n--- テスト開始 ---")
        test_dataset = UnifiedSkeletonDataset(test_data_path, test_label_path, original_to_mapped_map, category_map, NON_MG_MAPPED_LABEL, use_bone=use_bone,
            use_delta_joint=use_delta_joint, use_delta_bone=use_delta_bone)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        model_path = os.path.join(model_save_dir, 'best_unified_model.pth')
        if not os.path.exists(model_path): model_path = os.path.join(model_save_dir, 'epoch_10_unified_model.pth')
        print(f"テストモデルをロード: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_unified_model(model, test_loader, device, mapped_to_original_map, 'test_results_ensemble.txt')
    else:
        print("\n--- テストはスキップされました ---")