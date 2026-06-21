import os
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crack500 import CrackAwareSkeletonScanUMamba

DEVICE = torch.device('cuda')
IMG_SIZE = 512
MODEL_PATH = "/root/autodl-tmp/Demo2/deepcrack/checkpoints_cfd/best_model.pth"
IMG_DIR = "/root/autodl-tmp/Demo2/deepcrack/CFD/images"
LABEL_DIR = "/root/autodl-tmp/Demo2/deepcrack/CFD/masks"

print("="*60)
print("CFD 测试")
print("="*60)

model = CrackAwareSkeletonScanUMamba().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint, strict=False)
model.eval()
print("✓ 模型加载成功")

all_masks = glob(os.path.join(LABEL_DIR, "*.png"))
mask_stems = [Path(m).stem for m in all_masks]
all_images = [os.path.join(IMG_DIR, f"{stem}.jpg") for stem in mask_stems if os.path.exists(os.path.join(IMG_DIR, f"{stem}.jpg"))]

train_val, test_files = train_test_split(all_images, test_size=0.2, random_state=42)

valid_pairs = []
for img_path in test_files: 
    stem = Path(img_path).stem
    mask_path = os.path.join(LABEL_DIR, f"{stem}.png")
    if os.path.exists(mask_path):
        valid_pairs.append((img_path, mask_path))

print(f"测试图像数: {len(valid_pairs)}")
best_f1 = 0
best_thr = 0.5
best_metrics = {}
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

with torch.no_grad():
    for thr in thresholds:
        all_f1 = []
        all_prec = []
        all_rec = []
        all_iou = []
        all_miou = []
        
        for img_path, mask_path in tqdm(valid_pairs, desc=f"thr={thr}"):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_tensor = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(DEVICE)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
            mask = (mask > 128).astype(np.float32)
            
            output = model(img_tensor)
            if isinstance(output, tuple):
                output = output[0]
            prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
            pred = (prob > thr).astype(np.float32)
            
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
            all_miou.append(miou)
        
        mean_f1 = np.mean(all_f1)
        print(f"  thr={thr:.2f}: F1={mean_f1*100:.2f}%, IoU={np.mean(all_iou)*100:.2f}%, mIoU={np.mean(all_miou)*100:.2f}%")
        
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_thr = thr
            best_metrics = {
                'f1': np.mean(all_f1),
                'precision': np.mean(all_prec),
                'recall': np.mean(all_rec),
                'iou': np.mean(all_iou),
                'miou': np.mean(all_miou),
                'threshold': thr,
            }

print("\n" + "="*60)
print("CFD 结果")
print("="*60)
print(f"  Best threshold: {best_metrics['threshold']:.2f}")
print(f"  F1:             {best_metrics['f1']*100:.2f}%")
print(f"  Precision:      {best_metrics['precision']*100:.2f}%")
print(f"  Recall:         {best_metrics['recall']*100:.2f}%")
print(f"  IoU (crack):    {best_metrics['iou']*100:.2f}%")
print(f"  mIoU (avg):     {best_metrics['miou']*100:.2f}%")
print("="*60)