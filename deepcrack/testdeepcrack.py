import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crack500 import CrackAwareSkeletonScanUMamba

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512

MODEL_PATH = "/root/autodl-tmp/Demo2/deepcrack/checkpoints_deepcrack/best_model.pth"

IMG_DIR = "/root/autodl-tmp/Demo2/DeepCrack/test_img"
MASK_DIR = "/root/autodl-tmp/Demo2/DeepCrack/test_lab"


def predict_with_tta(model, image):
    """ms_flip TTA"""
    B, C, H, W = image.shape
    all_preds = []
    scales = [0.75, 1.0, 1.25]
    
    for scale in scales:
        if scale == 1.0:
            scaled = image
        else:
            scaled = F.interpolate(image, scale_factor=scale, mode="bilinear", align_corners=False)
        
        # 原始
        output = model(scaled)
        if isinstance(output, tuple):
            output = output[0]
        prob = torch.softmax(output, dim=1)[:, 1, :, :]
        prob = F.interpolate(prob.unsqueeze(1), size=(H, W), mode="bilinear").squeeze(1)
        all_preds.append(prob)
        
        # 水平翻转
        flipped = torch.flip(scaled, dims=[3])
        output = model(flipped)
        if isinstance(output, tuple):
            output = output[0]
        prob = torch.flip(torch.softmax(output, dim=1)[:, 1, :, :], dims=[2])
        prob = F.interpolate(prob.unsqueeze(1), size=(H, W), mode="bilinear").squeeze(1)
        all_preds.append(prob)
        
        # 垂直翻转
        flipped = torch.flip(scaled, dims=[2])
        output = model(flipped)
        if isinstance(output, tuple):
            output = output[0]
        prob = torch.flip(torch.softmax(output, dim=1)[:, 1, :, :], dims=[1])
        prob = F.interpolate(prob.unsqueeze(1), size=(H, W), mode="bilinear").squeeze(1)
        all_preds.append(prob)
    
    return torch.stack(all_preds).mean(dim=0)


def test_deepcrack():
    print("="*60)
    print("DeepCrack测试 (含 mIoU)")
    print("="*60)
    
    # 检查路径
    if not os.path.exists(IMG_DIR):
        print(f"错误: 图像目录不存在 - {IMG_DIR}")
        return
    if not os.path.exists(MASK_DIR):
        print(f"错误: Mask目录不存在 - {MASK_DIR}")
        return
    
    print(f"Image dir: {IMG_DIR}")
    print(f"Mask dir: {MASK_DIR}")
    
    # 收集有效图像
    img_files = sorted(glob(os.path.join(IMG_DIR, "*.jpg")) + glob(os.path.join(IMG_DIR, "*.png")))
    valid_pairs = []
    for img_path in img_files:
        stem = Path(img_path).stem
        mask_path = os.path.join(MASK_DIR, f"{stem}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(MASK_DIR, f"{stem}.bmp")
        if os.path.exists(mask_path):
            valid_pairs.append((img_path, mask_path))
    
    print(f"有效测试图像数: {len(valid_pairs)}")
    
    if len(valid_pairs) == 0:
        print("错误: 未找到有效的图像-mask对！")
        return
    
    # 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 - {MODEL_PATH}")
        return
    
    model = CrackAwareSkeletonScanUMamba().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✓ 模型加载成功")
    
    # 阈值搜索
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    best_f1 = 0
    best_thr = 0.5
    best_metrics = {}
    
    print("\n开始测试...")
    with torch.no_grad():
        for thr in thresholds:
            all_f1 = []
            all_prec = []
            all_rec = []
            all_iou = []
            all_miou = []
            all_iou_bg = []
            
            for img_path, mask_path in tqdm(valid_pairs, desc=f"thr={thr}"):
                # 读取图像
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img_tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
                
                # 读取mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
                mask = (mask > 128).astype(np.float32)
                
                # TTA推理
                prob = predict_with_tta(model, img_tensor)
                pred = (prob > thr).float().cpu().numpy()
                
                tp = (pred * mask).sum()
                fp = (pred * (1 - mask)).sum()
                fn = ((1 - pred) * mask).sum()
                tn = ((1 - pred) * (1 - mask)).sum()
                eps = 1e-7
                
                prec = tp / (tp + fp + eps)
                rec = tp / (tp + fn + eps)
                f1 = 2 * prec * rec / (prec + rec + eps)
                iou = tp / (tp + fp + fn + eps)
                iou_bg = tn / (tn + fp + fn + eps)
                miou = (iou + iou_bg) / 2
                
                all_f1.append(f1)
                all_prec.append(prec)
                all_rec.append(rec)
                all_iou.append(iou)
                all_iou_bg.append(iou_bg)
                all_miou.append(miou)
            
            mean_f1 = np.mean(all_f1)
            mean_iou = np.mean(all_iou)
            mean_miou = np.mean(all_miou)
            mean_iou_bg = np.mean(all_iou_bg)
            
            print(f"  thr={thr:.2f}: F1={mean_f1*100:.2f}%, IoU(crack)={mean_iou*100:.2f}%, mIoU={mean_miou*100:.2f}%")
            
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_thr = thr
                best_metrics = {
                    'f1': mean_f1,
                    'precision': np.mean(all_prec),
                    'recall': np.mean(all_rec),
                    'iou': np.mean(all_iou),
                    'iou_bg': np.mean(all_iou_bg),
                    'miou': np.mean(all_miou),
                    'threshold': thr,
                }
    
    print("\n" + "="*60)
    print("DeepCrack 测试结果")
    print("="*60)
    print(f"  最佳阈值:           {best_metrics['threshold']:.2f}")
    print(f"  F1 (Dice):          {best_metrics['f1']*100:.2f}%")
    print(f"  Precision:          {best_metrics['precision']*100:.2f}%")
    print(f"  Recall:             {best_metrics['recall']*100:.2f}%")
    print(f"  IoU (crack class):  {best_metrics['iou']*100:.2f}%")
    print(f"  IoU (background):   {best_metrics['iou_bg']*100:.2f}%")
    print(f"  mIoU (all classes): {best_metrics['miou']*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    test_deepcrack()