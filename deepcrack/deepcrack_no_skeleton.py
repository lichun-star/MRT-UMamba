import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from glob import glob
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.UMambaBot_2d import UMambaBot
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
                alpha = np.random.uniform(0.8, 1.3)
                beta = np.random.randint(-20, 20)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if np.random.random() > 0.7:
                noise = np.random.normal(0, 3, img.shape).astype(np.uint8)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()


class CoarseFineLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        if logits.shape[1] > 1:
            prob = torch.softmax(logits, dim=1)[:, 1]
            logit = logits[:, 1]
        else:
            prob = torch.sigmoid(logits[:, 0])
            logit = logits[:, 0]
        target = target.float()
        intersection = (prob * target).sum()
        dice = (2.0 * intersection + self.smooth) / (prob.sum() + target.sum() + self.smooth)
        dice_loss = 1.0 - dice
        bce_loss = F.binary_cross_entropy_with_logits(logit, target)
        return dice_loss + bce_loss


# ==================== 模型组件 ====================
class SobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx*gx + gy*gy + 1e-6)


class PyramidContext(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels,8,1,bias=False), nn.BatchNorm2d(8), nn.ReLU(True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels,8,1,bias=False), nn.BatchNorm2d(8), nn.ReLU(True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels,8,1,bias=False), nn.BatchNorm2d(8), nn.ReLU(True))
        self.branch4 = nn.Sequential(nn.Conv2d(in_channels,8,1,bias=False), nn.BatchNorm2d(8), nn.ReLU(True))
        self.fuse = nn.Sequential(
            nn.Conv2d(32, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def _pool(self, x, size):
        p = F.adaptive_avg_pool2d(x, size)
        return F.interpolate(p, size=x.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        b1 = self.branch1(self._pool(x,1))
        b2 = self.branch2(self._pool(x,2))
        b3 = self.branch3(self._pool(x,4))
        b4 = self.branch4(self._pool(x,8))
        return self.fuse(torch.cat([b1,b2,b3,b4], dim=1))


class CrackAwareScanMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.direction_predictor = nn.Sequential(
            nn.Conv2d(dim, dim//4, 3, padding=1),
            nn.BatchNorm2d(dim//4),
            nn.ReLU(True),
            nn.Conv2d(dim//4, 8, 1),
        )
        self.fine_crack_detector = nn.Sequential(
            nn.Conv2d(dim, dim//4, 3, padding=1),
            nn.BatchNorm2d(dim//4),
            nn.ReLU(True),
            nn.Conv2d(dim//4, 1, 1),
            nn.Sigmoid(),
        )
        self.direction_weights = nn.Parameter(torch.ones(8)/8)

    def forward(self, x):
        dir_logits = self.direction_predictor(x)
        dir_probs = torch.softmax(dir_logits, dim=1)
        fine_weight = self.fine_crack_detector(x)
        weighted_dir = (dir_probs * self.direction_weights.view(1,-1,1,1)).sum(dim=1, keepdim=True)
        enhanced = x + x*fine_weight*0.3 + x*weighted_dir*0.2
        return enhanced, fine_weight


class NoSkeletonUMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = UMambaBot(
            input_channels=3,
            n_stages=5,
            features_per_stage=[32,64,128,256,320],
            conv_op=nn.Conv2d,
            kernel_sizes=[[3,3]]*5,
            strides=[[1,1],[2,2],[2,2],[2,2],[2,2]],
            n_conv_per_stage=2,
            num_classes=2,
            n_conv_per_stage_decoder=2,
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={'eps':1e-5,'affine':True},
            dropout_op=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'negative_slope':0.01,'inplace':True},
            deep_supervision=False,
        )
        
        self.edge = SobelEdge()
        self.global_ctx = PyramidContext(in_channels=1, out_channels=16)
        self.global_head = nn.Sequential(
            nn.Conv2d(16,16,3,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 2, 1),
        )
        self.coarse_fuse = nn.Sequential(
            nn.Conv2d(21, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.scan_enhancer = CrackAwareScanMamba(dim=32)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(32,16,3,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16,1,1),
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(32,32,3,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 2, 1),
        )
        self.uncertain_gate_proj = nn.Sequential(
            nn.Conv2d(1,8,3,padding=1,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8,1,1),
        )

    def forward(self, x):
        coarse_logits = self.base_model(x)
        
        if coarse_logits.shape[1] > 1:
            probs = torch.softmax(coarse_logits, dim=1)
            crack_prob = probs[:, 1:2]
            bg_prob = probs[:, 0:1]
        else:
            crack_prob = torch.sigmoid(coarse_logits)
            bg_prob = 1.0 - crack_prob
        
        entropy = -(crack_prob * torch.log(crack_prob + 1e-6) + bg_prob * torch.log(bg_prob + 1e-6))
        edge_mag = self.edge(crack_prob)
        uncertainty = 4.0 * crack_prob * (1.0 - crack_prob)
        uncertainty_gate = torch.sigmoid(self.uncertain_gate_proj(uncertainty))
        global_feat = self.global_ctx(crack_prob)
        
        coarse_feat = torch.cat([crack_prob, bg_prob, entropy, edge_mag, uncertainty, global_feat], dim=1)
        coarse_feat = self.coarse_fuse(coarse_feat)
        scan_feat, fine_weight = self.scan_enhancer(coarse_feat)
        
        boundary_logits = self.boundary_head(scan_feat)
        boundary_prob = torch.sigmoid(boundary_logits)
        residual = self.refine_head(scan_feat)
        global_logits = self.global_head(global_feat)
        global_prob = torch.sigmoid(global_logits)
        
        refined_logits = coarse_logits + residual * (0.25 + 0.75 * boundary_prob)
        refined_logits = refined_logits + 0.08 * residual * uncertainty_gate
        refined_logits = refined_logits + 0.05 * residual * fine_weight
        refined_logits = refined_logits + 0.10 * global_logits * global_prob
        refined_logits = refined_logits - 0.06 * residual * (1.0 - boundary_prob)
        
        return refined_logits


def evaluate(model, loader, device):
    model.eval()
    best_f1 = 0
    best_thr = 0.5
    best_metrics = {}
    
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    
    for thr in thresholds:
        total_tp, total_fp, total_fn = 0.0, 0.0, 0.0
        
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                outputs = model(imgs)
            
            if outputs.shape[1] > 1:
                probs = torch.softmax(outputs, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(outputs[:, 0])
            
            pred = (probs > thr).float()
            target = masks.float()
            
            total_tp += (pred * target).sum().item()
            total_fp += (pred * (1 - target)).sum().item()
            total_fn += ((1 - pred) * target).sum().item()
        
        eps = 1e-7
        precision = total_tp / (total_tp + total_fp + eps)
        recall = total_tp / (total_tp + total_fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_metrics = {'f1': f1, 'precision': precision, 'recall': recall, 'threshold': thr}
    
    return best_metrics


def load_pretrained(model, pretrained_path, device):
    if os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('base_model.'):
                new_k = k[11:]
            else:
                new_k = k
            new_state_dict[new_k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f"从 {pretrained_path} 加载预训练权重")
        return True
    return False


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
    
    model = NoSkeletonUMamba().to(device)
    
    pretrained_path = "/root/autodl-tmp/Demo2/deepcrack/checkpoints_crackaware_skeleton_scan/best_model.pth"
    load_pretrained(model, pretrained_path, device)
    
    optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=CONFIG['lr'], rho=CONFIG['rho'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = CoarseFineLoss()
    
    save_dir = "/root/autodl-tmp/Demo2/deepcrack/checkpoints_deepcrack_no_skeleton_transfer"
    os.makedirs(save_dir, exist_ok=True)
    
    best_f1 = 0
    patience_counter = 0
    
    print("\n开始训练 No Skeleton on DeepCrack...")
    
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
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val F1={metrics['f1']*100:.2f}%, thr={metrics['threshold']:.2f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            patience_counter = 0
            print(f"  ✓ 保存最佳模型 (F1={metrics['f1']*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"早停于 epoch {epoch+1}")
                break
    
    print("\n加载最佳模型测试...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    test_metrics = evaluate(model, test_loader, device)
    
    print("\n" + "=" * 60)
    print("Exp-C: No Skeleton on DeepCrack")
    print("=" * 60)
    print(f"  F1:        {test_metrics['f1']*100:.2f}%")
    print(f"  Precision: {test_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {test_metrics['recall']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    import torch.nn.functional as F
    main()