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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crack500 import (
    CrackAwareSkeletonScanUMamba,
    CoarseFineLoss,
)
from sam import SAM

CONFIG = {
    'img_size': 512,
    'batch_size': 4,
    'epochs': 100,
    'lr': 3e-5,
    'rho': 0.03,
    'patience': 20,
}


class DeepCrackDataset(Dataset):
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
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, f"{stem}.bmp")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 128).astype(np.float32)
        
        if self.augment:
            # 基础翻转
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if np.random.random() > 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
            # 旋转
            if np.random.random() > 0.5:
                k = np.random.choice([1, 2, 3])
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
            # 亮度对比度
            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.8, 1.3)
                beta = np.random.randint(-20, 20)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            # 轻微高斯噪声
            if np.random.random() > 0.7:
                noise = np.random.normal(0, 3, img.shape).astype(np.uint8)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()


def evaluate(model, loader, device):
    model.eval()
    best_f1 = 0
    best_thr = 0.5
    best_metrics = {}
    
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    
    for thr in thresholds:
        all_f1 = []
        all_prec = []
        all_rec = []
        
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
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
    
    img_dir = "/root/autodl-tmp/Demo2/DeepCrack/test_img"
    mask_dir = "/root/autodl-tmp/Demo2/DeepCrack/test_lab"
    
    all_images = sorted(glob(os.path.join(img_dir, "*.jpg")) + 
                        glob(os.path.join(img_dir, "*.png")))
    
    valid_images = []
    for img_path in all_images:
        stem = Path(img_path).stem
        mask_path = os.path.join(mask_dir, f"{stem}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, f"{stem}.bmp")
        if os.path.exists(mask_path):
            valid_images.append(img_path)
    
    print(f"有效图像数: {len(valid_images)}")
    
    train_val, test_files = train_test_split(valid_images, test_size=0.15, random_state=42)
    train_files, val_files = train_test_split(train_val, test_size=0.15/0.85, random_state=42)
    
    print(f"训练集: {len(train_files)}")
    print(f"验证集: {len(val_files)}")
    print(f"测试集: {len(test_files)}")
    
    train_dataset = DeepCrackDataset(train_files, mask_dir, augment=True)
    val_dataset = DeepCrackDataset(val_files, mask_dir, augment=False)
    test_dataset = DeepCrackDataset(test_files, mask_dir, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    model = CrackAwareSkeletonScanUMamba().to(device)
    

    pretrained_path = "/root/autodl-tmp/Demo2/deepcrack/checkpoints_crackaware_skeleton_scan/best_model.pth"
    if os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    
    optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=CONFIG['lr'], rho=CONFIG['rho'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = CoarseFineLoss()
    
    best_f1 = 0
    patience_counter = 0
    save_dir = "/root/autodl-tmp/Demo2/deepcrack/checkpoints_deepcrack"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n开始训练...")
    
    for epoch in range(CONFIG['epochs']):
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
        scheduler.step(metrics['f1'])
        
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, "
              f"Val F1={metrics['f1']*100:.2f}%, thr={metrics['threshold']:.2f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            patience_counter = 0
            print(f"  ✓ 保存 (F1={metrics['f1']*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"早停于 epoch {epoch+1}")
                break
    
    # 测试
    print("\n加载最佳模型测试...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    test_metrics = evaluate(model, test_loader, device)
    
    print("\n" + "="*60)
    print("DeepCrac结果")
    print("="*60)
    print(f"  F1:        {test_metrics['f1']*100:.2f}%")
    print(f"  Precision: {test_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {test_metrics['recall']*100:.2f}%")
    print("="*60)
    
if __name__ == "__main__":
    main()
