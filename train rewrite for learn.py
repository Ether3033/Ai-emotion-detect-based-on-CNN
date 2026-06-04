# train.py
import os
import time
import json
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

# 导入自定义模块（假设 train.py 与 dataset.py、models.py 在同一目录）
from dataset import get_dataloaders, FER2013Dataset
from models import ABC_Forward


# ------------------------- 训练/验证函数 -------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / total
    val_acc = correct / total
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    return val_loss, val_acc, val_f1, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ------------------------- 绘图函数 -------------------------
def plot_confusion_matrix(all_labels, all_preds, class_names):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def plot_roc_curves(all_labels, all_probs, class_names):
    """绘制多分类 ROC 曲线（One-vs-Rest）"""
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    n_classes = len(class_names)
    y_bin = label_binarize(all_labels, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right', fontsize='small')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


# ------------------------- 日志管理器 -------------------------
class TrainingLogger:
    """训练日志管理器，支持实时写入和保存"""
    def __init__(self, model_name, log_dir='logs'):
        self.model_name = model_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.logs = []
        self.start_time = None

    def log_start(self, info):
        self.start_time = time.time()
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'start',
            'info': info
        }
        self.logs.append(entry)
        self._save()
        return f"[{datetime.now().strftime('%H:%M:%S')}] 训练开始: {info}"

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, val_f1):
        elapsed = time.time() - self.start_time
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'epoch',
            'epoch': epoch,
            'train_loss': round(float(train_loss), 4),
            'train_acc': round(float(train_acc), 4),
            'val_loss': round(float(val_loss), 4),
            'val_acc': round(float(val_acc), 4),
            'val_f1': round(float(val_f1), 4),
            'elapsed_seconds': round(elapsed, 1)
        }
        self.logs.append(entry)
        self._save()
        return (f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f} | Time: {elapsed:.1f}s")

    def log_message(self, message):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'message',
            'message': message
        }
        self.logs.append(entry)
        self._save()
        return f"[{datetime.now().strftime('%H:%M:%S')}] {message}"

    def _save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)


# ------------------------- 训练主流程（生成器） -------------------------
def train_model(model_name, csv_path, device, batch_size=64, epochs=30,
                lr=1e-3, weight_decay=1e-4, patience=7, num_workers=2):
    """
    完整训练流程（生成器形式，支持实时更新）
    """
    # 加载数据（使用 dataset.py 中的 get_dataloaders）
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=csv_path,
        batch_size=batch_size,
        num_workers=num_workers
    )
    class_names = FER2013Dataset.EMOTION_NAMES

    # 创建模型（使用函数工厂）
    model = ABC_Forward(model_name, num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    logger = TrainingLogger(model_name)
    yield {'type': 'log', 'message': logger.log_start(
        f'Model: {model_name}, Epochs: {epochs}, LR: {lr}, BatchSize: {batch_size}')}

    best_val_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    stopped_early = False

    # 创建保存目录
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        log_msg = logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, val_f1)

        # 绘图
        cm_img = plot_confusion_matrix(val_labels, val_preds, class_names)
        roc_img = plot_roc_curves(val_labels, val_probs, class_names)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pt')
            cm_img.save(f'checkpoints/{model_name}_best_cm.png')
            roc_img.save(f'checkpoints/{model_name}_best_roc.png')
        else:
            no_improve += 1

        if no_improve >= patience:
            stopped_early = True

        result = {
            'type': 'epoch_result',
            'epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'no_improve': no_improve,
            'log_message': log_msg,
            'cm_image': cm_img,
            'roc_image': roc_img,
            'stopped_early': stopped_early,
            'is_best': is_best,
        }
        yield result

        if stopped_early:
            yield {'type': 'log', 'message': logger.log_message(
                f'早停触发！最佳 val F1: {best_val_f1:.4f} (Epoch {best_epoch})')}
            break

    # 加载最佳模型，并评估测试集（可选）
    model.load_state_dict(torch.load(f'checkpoints/{model_name}_best.pt'))
    test_loss, test_acc, test_f1, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, device
    )
    yield {
        'type': 'complete',
        'message': f'训练完成！最佳 Val F1: {best_val_f1:.4f} (Epoch {best_epoch}) | Test F1: {test_f1:.4f}',
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch,
        'test_f1': test_f1,
    }


# ------------------------- 运行示例（可独立执行）-------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='FER2013 CSV 文件路径')
    parser.add_argument('--model', type=str, choices=['A', 'B', 'C'], default='B', help='选择模型类型')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # 训练模型（消费生成器，打印进度）
    for result in train_model(args.model, args.csv_path, device=args.device,
                              batch_size=args.batch_size, epochs=args.epochs, lr=args.lr):
        if result['type'] == 'epoch_result':
            print(result['log_message'])
        elif result['type'] == 'log':
            print(result['message'])
        elif result['type'] == 'complete':
            print(result['message'])