import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from glob import glob
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crack500 import (
    CrackAwareSkeletonScanUMamba,
    CoarseFineLoss,
)
from sam import SAM

CONFIG = {
    'img_size': 512,
    'batch_size': 4,
    'epochs': 150,
    'lr': 1e-4,
    'rho': 0.03,
}


class CFDDataset(Dataset):
    def __init__(self, img_files, mask_dir, img_size=512, augment=True):
        self.img_files = img_files
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        stem = Path(img_path).stem
        mask_path = os.path.join(self.mask_dir, f"{stem}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 128).astype(np.float32)
        
        if self.augment:
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if np.random.random() > 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
            if np.random.random() > 0.5:
                k = np.random.choice([1, 2, 3])
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()


def evaluate(model, loader, device):
    model.eval()
    best_f1 = 0
    best_thr = 0.5
    best_metrics = {}
    
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    
    for thr in thresholds:
        all_f1 = []
        all_prec = []
        all_rec = []
        
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                outputs = model(imgs)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            pred = (probs > thr).float()
            target = masks.float()
            
            tp = (pred * target).sum().item()
            fp = (pred * (1 - target)).sum().item()
            fn = ((1 - pred) * target).sum().item()
            eps = 1e-7
            
            f1 = 2 * tp / (2 * tp + fp + fn + eps)
            prec = tp / (tp + fp + eps)
            rec = tp / (tp + fn + eps)
            
            all_f1.append(f1)
            all_prec.append(prec)
            all_rec.append(rec)
        
        mean_f1 = np.mean(all_f1)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_thr = thr
            best_metrics = {
                'f1': mean_f1,
                'precision': np.mean(all_prec),
                'recall': np.mean(all_rec),
                'threshold': thr,
            }
    
    return best_metrics


def main():
    device = torch.device('cuda')
    print(f"设备: {device}")
    
    img_dir = "/root/autodl-tmp/Demo2/deepcrack/CFD/images"
    mask_dir = "/root/autodl-tmp/Demo2/deepcrack/CFD/masks"
    
    all_masks = glob(os.path.join(mask_dir, "*.png"))
    mask_stems = [Path(m).stem for m in all_masks]
    
    all_images = []
    for stem in mask_stems:
        img_path = os.path.join(img_dir, f"{stem}.jpg")
        if os.path.exists(img_path):
            all_images.append(img_path)
    
    print(f"有mask的图像数: {len(all_images)}")
    
    train_val, test_files = train_test_split(all_images, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_val, test_size=0.25, random_state=42)
    
    print(f"训练集: {len(train_files)}")
    print(f"验证集: {len(val_files)}")
    print(f"测试集: {len(test_files)}")
    
    train_dataset = CFDDataset(train_files, mask_dir, augment=True)
    val_dataset = CFDDataset(val_files, mask_dir, augment=False)
    test_dataset = CFDDataset(test_files, mask_dir, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    model = CrackAwareSkeletonScanUMamba().to(device)
    
    save_dir = "/root/autodl-tmp/Demo2/deepcrack/checkpoints_cfd"
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")
    history_path = os.path.join(save_dir, "training_history.json")
    
    start_epoch = 0
    best_f1 = 0.0
    
    history = {'epochs': [], 'val_f1': [], 'best_f1': 0.0}
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        if history['epochs']:
            start_epoch = max(history['epochs']) + 1
            best_f1 = history.get('best_f1', 0.0)
            print(f"从训练历史恢复:")
            print(f"  - 已训练轮数: {len(history['epochs'])}")
            print(f"  - 起始 Epoch: {start_epoch + 1}")
            print(f"  - 最佳 F1: {best_f1*100:.2f}%")
    
    # 加载模型权重（优先使用最佳模型）
    if os.path.exists(best_model_path):
        print(f"\n加载最佳模型: {best_model_path}")
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ 已加载最佳模型 (F1={best_f1*100:.2f}%)")
    else:
        pretrained_path = "/root/autodl-tmp/Demo2/deepcrack/checkpoints_crackaware_skeleton_scan/best_model.pth"
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=device)
            # ========== 修复：去除 base_model. 前缀 ==========
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('base_model.'):
                    new_k = k[11:]
                else:
                    new_k = k
                new_state_dict[new_k] = v
            # ==============================================
            model.load_state_dict(new_state_dict, strict=False)
    
    optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=CONFIG['lr'], rho=CONFIG['rho'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    if os.path.exists(checkpoint_path):
        print(f"\n发现完整检查点，加载优化器状态...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ 已恢复优化器和调度器状态")
        except:
            print("⚠ 优化器状态加载失败，使用新初始化的优化器")
    
    criterion = CoarseFineLoss()
    
    print(f"\n开始训练... (共 {CONFIG['epochs']} 轮)")
    if start_epoch > 0:
        print(f"从 Epoch {start_epoch + 1} 继续训练")
    else:
        print("从头开始训练")
    
    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.first_step(zero_grad=True)
            outputs2 = model(imgs)
            loss2 = criterion(outputs2, masks)
            loss2.backward()
            optimizer.second_step(zero_grad=True)
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        metrics = evaluate(model, val_loader, device)
        current_f1 = metrics['f1']
        scheduler.step(current_f1)
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
              f"Val F1={current_f1*100:.2f}%, thr={metrics['threshold']:.2f}")
        
        history['epochs'].append(epoch)
        history['val_f1'].append(current_f1)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), best_model_path)
            history['best_f1'] = best_f1
            print(f"  ✓ 保存最佳模型 (F1={current_f1*100:.2f}%)")
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimsave_dirizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
            'val_f1': current_f1,
        }
        torch.save(checkpoint, checkpoint_path)
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\n训练完成，已删除中断点文件")
    
    print("\n加载最佳模型测试...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("警告: 未找到最佳模型文件，使用当前模型测试")
    
    test_metrics = evaluate(model, test_loader, device)
    
    print("\n" + "="*60)
    print("CFD 结果")
    print("="*60)
    print(f"  总训练轮数: {len(history['epochs'])}")
    print(f"  最佳验证 F1: {history['best_f1']*100:.2f}%")
    print(f"  测试集 F1:   {test_metrics['f1']*100:.2f}%")
    print(f"  Precision:   {test_metrics['precision']*100:.2f}%")
    print(f"  Recall:      {test_metrics['recall']*100:.2f}%")
    print(f"  最佳阈值:    {test_metrics['threshold']:.2f}")
    print("="*60)
    
    final_results = {
        'best_val_f1': history['best_f1'],
        'test_f1': test_metrics['f1'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'threshold': test_metrics['threshold'],
        'total_epochs': len(history['epochs'])
    }
    with open(os.path.join(save_dir, "final_results.json"), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n结果已保存到: {save_dir}")


if __name__ == "__main__":
    main()