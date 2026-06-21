import os
import sys
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.UMambaBot_2d import UMambaBot
from sam import SAM


CONFIG = {
    'data_path': '/root/autodl-tmp/Demo2/deepcrack/CRACK500',
    'save_dir': '/root/autodl-tmp/Demo2/deepcrack/checkpoints_crackaware_skeleton_scan',
    'resume_path': '',
    'img_size': 512,
    'batch_size': 2,
    'total_epochs': 60,
    'lr': 1e-4,
    'rho': 0.03,
    'num_classes': 2,
    'input_channels': 3,
    'warmup_epochs': 6,
    'threshold_min': 0.30,
    'threshold_max': 0.70,
    'threshold_steps': 9,
    'coarse_weight': 0.35,
    'fine_weight': 1.00,
    'boundary_weight': 0.10,
    'skeleton_weight': 0.16,
    'uncertain_weight': 0.18,
    'consistency_weight': 0.06,
    'gate_weight': 0.03,
    'bridge_weight': 0.04,
}


class Crack500Dataset(Dataset):
    def __init__(self, data_path, split='train', img_size=512, augment=True):
        self.img_size = img_size
        self.augment = augment
        self.split = split

        if split == 'train':
            self.img_dir = os.path.join(data_path, 'train_images_512')
            self.mask_dir = os.path.join(data_path, 'train_labels_512')
        else:
            self.img_dir = os.path.join(data_path, 'test_images_512')
            self.mask_dir = os.path.join(data_path, 'test_labels_512')

        import glob
        self.img_files = sorted(
            glob.glob(os.path.join(self.img_dir, '*.jpg')) +
            glob.glob(os.path.join(self.img_dir, '*.png'))
        )
        print(f"{split}: 找到 {len(self.img_files)} 张图像")

        self.sample_weights = None
        if split == 'train':
            self.sample_weights = self._build_sample_weights()

    def _mask_path_from_img(self, img_path):
        base = os.path.basename(img_path)
        name = os.path.splitext(base)[0]
        return os.path.join(self.mask_dir, f'{name}.png')

    def _build_sample_weights(self):
        weights = []
        for img_path in self.img_files:
            mask_path = self._mask_path_from_img(img_path)
            fg_ratio = 0.0
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, (self.img_size, self.img_size))
                    fg_ratio = float((mask > 128).mean())

            if fg_ratio < 1e-4:
                weight = 1.0
            elif fg_ratio < 0.002:
                weight = 2.5
            elif fg_ratio < 0.01:
                weight = 3.5
            else:
                weight = 4.0
            weights.append(weight)
        return torch.as_tensor(weights, dtype=torch.double)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'无法读取图像: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        mask_path = self._mask_path_from_img(img_path)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = (mask > 128).astype(np.float32)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        if self.augment:
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if np.random.random() > 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
            if np.random.random() > 0.3:
                alpha = 0.9 + np.random.random() * 0.25
                beta = np.random.randint(-12, 12)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()


class BoundaryTargetBuilder:
    def __init__(self, kernel_size=3):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __call__(self, mask_np):
        mask_u8 = (mask_np > 0.5).astype(np.uint8)
        boundary = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, self.kernel)
        return (boundary > 0).astype(np.float32)


class SkeletonTargetBuilder:
    def __init__(self):
        self.element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    def __call__(self, mask_np):
        img = (mask_np > 0.5).astype(np.uint8)
        skel = np.zeros_like(img)
        while True:
            eroded = cv2.erode(img, self.element)
            opened = cv2.dilate(eroded, self.element)
            temp = cv2.subtract(img, opened)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return (skel > 0).astype(np.float32)


class CoarseFineLoss(nn.Module):
    def __init__(self, smooth=1e-5, pos_weight=None):
        super().__init__()
        self.smooth = smooth
        self.pos_weight = pos_weight

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

        if self.pos_weight is None:
            bce_loss = F.binary_cross_entropy_with_logits(logit, target)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                logit,
                target,
                pos_weight=self.pos_weight.to(logit.device)
            )
        return dice_loss + bce_loss


class SobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)


class PyramidContext(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(nn.Conv2d(in_channels, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(32, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _pool(self, x, size):
        pooled = F.adaptive_avg_pool2d(x, size)
        return F.interpolate(pooled, size=x.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        b1 = self.branch1(self._pool(x, 1))
        b2 = self.branch2(self._pool(x, 2))
        b3 = self.branch3(self._pool(x, 4))
        b4 = self.branch4(self._pool(x, 8))
        return self.fuse(torch.cat([b1, b2, b3, b4], dim=1))


class CrackAwareScanMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.direction_predictor = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 8, kernel_size=1),
        )
        self.fine_crack_detector = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.direction_weights = nn.Parameter(torch.ones(8) / 8)

    def forward(self, x):
        direction_logits = self.direction_predictor(x)
        direction_probs = torch.softmax(direction_logits, dim=1)
        fine_weight = self.fine_crack_detector(x)
        weighted_dir = (direction_probs * self.direction_weights.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        enhanced = x + x * fine_weight * 0.3 + x * weighted_dir * 0.2
        return enhanced, fine_weight


class ScanEnhancedCoarseToFineRefiner(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.edge = SobelEdge()
        self.global_ctx = PyramidContext(in_channels=1, out_channels=16)
        self.global_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )
        self.coarse_fuse = nn.Sequential(
            nn.Conv2d(21, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.scan_enhancer = CrackAwareScanMamba(dim=32)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.skeleton_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )
        self.uncertain_gate_proj = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, coarse_logits):
        if coarse_logits.shape[1] > 1:
            print('-----------------------------------')
            print(coarse_logits.shape)
            crack_prob = torch.softmax(coarse_logits, dim=1)[:, 1:2]
            bg_prob = torch.softmax(coarse_logits, dim=1)[:, 0:1]
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
        skeleton_logits = self.skeleton_head(scan_feat)
        skeleton_prob = torch.sigmoid(skeleton_logits)
        residual = self.refine_head(scan_feat)
        global_logits = self.global_head(global_feat)
        global_prob = torch.sigmoid(global_logits)

        refined_logits = coarse_logits + residual * (0.25 + 0.75 * boundary_prob)
        refined_logits = refined_logits + 0.10 * skeleton_logits * skeleton_prob
        refined_logits = refined_logits + 0.08 * residual * uncertainty_gate
        refined_logits = refined_logits + 0.05 * residual * fine_weight
        refined_logits = refined_logits + 0.10 * global_logits * global_prob
        refined_logits = refined_logits - 0.06 * residual * (1.0 - boundary_prob)

        return refined_logits, boundary_logits, skeleton_logits, uncertainty, uncertainty_gate


class CrackAwareSkeletonScanUMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = UMambaBot(
            input_channels=CONFIG['input_channels'],
            n_stages=5,
            features_per_stage=[32, 64, 128, 256, 320],
            conv_op=nn.Conv2d,
            kernel_sizes=[[3, 3]] * 5,
            strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]],
            n_conv_per_stage=2,
            num_classes=CONFIG['num_classes'],
            n_conv_per_stage_decoder=2,
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'negative_slope': 0.01, 'inplace': True},
            deep_supervision=False,
        )
        self.refiner = ScanEnhancedCoarseToFineRefiner(num_classes=CONFIG['num_classes'])

    def forward(self, x, return_aux=False):
        coarse_logits = self.base_model(x)
        refined_logits, boundary_logits, skeleton_logits, uncertainty, uncertainty_gate = self.refiner(coarse_logits)
        if return_aux:
            return refined_logits, boundary_logits, skeleton_logits, coarse_logits, uncertainty, uncertainty_gate
        return refined_logits


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ['state_dict', 'model_state_dict', 'net', 'model']:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj if isinstance(obj, dict) else obj


def _strip_prefix(state_dict, prefixes=('module.', 'base_model.', 'refiner.')):
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
        cleaned[new_k] = v
    return cleaned


def load_partial_weights(module, state_dict, name='module'):
    module_sd = module.state_dict()
    loaded = {}
    for k, v in state_dict.items():
        if k in module_sd and v.shape == module_sd[k].shape:
            loaded[k] = v
    module_sd.update(loaded)
    module.load_state_dict(module_sd)
    print(f'{name}: 载入 {len(loaded)}/{len(state_dict)} 个匹配权重')


def update_ema_model(ema_model, model, decay=0.995):
    with torch.no_grad():
        ema_state = ema_model.state_dict()
        model_state = model.state_dict()
        for key, value in model_state.items():
            if key in ema_state and ema_state[key].dtype.is_floating_point:
                ema_state[key].mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                ema_state[key] = value.detach().clone()
        ema_model.load_state_dict(ema_state)


def create_model(device):
    model = CrackAwareSkeletonScanUMamba()
    if os.path.exists(CONFIG['resume_path']):
        print(f'加载预训练权重: {CONFIG["resume_path"]}')
        ckpt = torch.load(CONFIG['resume_path'], map_location=device)
        raw = _extract_state_dict(ckpt)
        raw = _strip_prefix(raw)

        base_sd = model.base_model.state_dict()
        base_loaded = {k: v for k, v in raw.items() if k in base_sd and v.shape == base_sd[k].shape}
        base_sd.update(base_loaded)
        model.base_model.load_state_dict(base_sd)
        print(f'base_model: 载入 {len(base_loaded)} 层权重')

        ref_sd = model.refiner.state_dict()
        ref_loaded = {k: v for k, v in raw.items() if k in ref_sd and v.shape == ref_sd[k].shape}
        ref_sd.update(ref_loaded)
        model.refiner.load_state_dict(ref_sd)
        if len(ref_loaded) > 0:
            print(f'refiner: 载入 {len(ref_loaded)} 层权重')
    return model


def compute_pos_weight(masks, floor=2.0, ceiling=8.0):
    fg = masks.float().sum().item()
    total = float(masks.numel())
    bg = max(total - fg, 1.0)
    ratio = bg / max(fg, 1.0)
    ratio = max(floor, min(ceiling, ratio))
    return torch.tensor(ratio, dtype=torch.float32)


@torch.no_grad()
def compute_metrics_from_logits(logits, masks, threshold=0.5):
    if logits.shape[1] > 1:
        probs = torch.softmax(logits, dim=1)[:, 1]
    else:
        probs = torch.sigmoid(logits[:, 0])
    pred = (probs > threshold).float()
    target = masks.float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def topk_uncertain_loss(logits, target, uncertainty, ratio=0.25):
    if logits.shape[1] > 1:
        logit = logits[:, 1]
    else:
        logit = logits[:, 0]
    target = target.float()
    bce_map = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    score = bce_map * (1.0 + 2.0 * uncertainty.squeeze(1))
    flat = score.reshape(-1)
    k = max(1, int(flat.numel() * ratio))
    return torch.topk(flat, k, largest=True).values.mean()


@torch.no_grad()
def predict_with_tta(model, imgs):
    logits = model(imgs)
    preds = [logits]

    hflip_imgs = torch.flip(imgs, dims=[3])
    hflip_logits = torch.flip(model(hflip_imgs), dims=[3])
    preds.append(hflip_logits)

    vflip_imgs = torch.flip(imgs, dims=[2])
    vflip_logits = torch.flip(model(vflip_imgs), dims=[2])
    preds.append(vflip_logits)

    hvflip_imgs = torch.flip(imgs, dims=[2, 3])
    hvflip_logits = torch.flip(model(hvflip_imgs), dims=[2, 3])
    preds.append(hvflip_logits)

    return torch.stack(preds, dim=0).mean(dim=0)


@torch.no_grad()
def evaluate_thresholds(model, loader, device):
    model.eval()
    thresholds = np.linspace(CONFIG['threshold_min'], CONFIG['threshold_max'], CONFIG['threshold_steps'])
    best = {'f1': -1.0, 'threshold': 0.5, 'precision': 0.0, 'recall': 0.0}

    for thr in thresholds:
        precisions, recalls, f1s = [], [], []
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = predict_with_tta(model, imgs)
            p, r, f = compute_metrics_from_logits(logits, masks, threshold=float(thr))
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        mean_f1 = float(np.mean(f1s))
        if mean_f1 > best['f1']:
            best = {
                'f1': mean_f1,
                'threshold': float(thr),
                'precision': float(np.mean(precisions)),
                'recall': float(np.mean(recalls)),
            }
    return best


def save_checkpoint(path, epoch, model, optimizer, best_f1, best_threshold, extra=None):
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': best_f1,
        'best_threshold': best_threshold,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    checkpoint_file = os.path.join(CONFIG['save_dir'], 'checkpoint.pth')
    best_model_file = os.path.join(CONFIG['save_dir'], 'best_model.pth')
    best_state_file = os.path.join(CONFIG['save_dir'], 'best_model_with_threshold.pth')
    start_epoch = 1
    best_f1 = 0.0
    best_threshold = 0.5

    train_dataset = Crack500Dataset(CONFIG['data_path'], 'train', CONFIG['img_size'], augment=True)
    val_dataset = Crack500Dataset(CONFIG['data_path'], 'test', CONFIG['img_size'], augment=False)
    boundary_builder = BoundaryTargetBuilder()
    skeleton_builder = SkeletonTargetBuilder()

    train_sampler = None
    if train_dataset.sample_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset.sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    print(f'训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}')

    model = create_model(device).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数量: {total_params / 1e6:.2f}M')
    print(f'可训练参数量: {trainable_params / 1e6:.2f}M')

    if os.path.exists(checkpoint_file):
        print('发现检查点，从中恢复...')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        if 'model_state_dict' in checkpoint:
            load_partial_weights(model, checkpoint['model_state_dict'], name='model')
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_f1 = float(checkpoint.get('best_f1', 0.0))
        best_threshold = float(checkpoint.get('best_threshold', 0.5))
        print(f'从 epoch {start_epoch} 继续训练，最佳 F1: {best_f1:.4f}, 最佳阈值: {best_threshold:.2f}')
    else:
        print('从头开始训练')

    optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=CONFIG['lr'], rho=CONFIG['rho'])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, CONFIG['total_epochs'] - start_epoch + 1))
    seg_loss_fn = CoarseFineLoss()

    print('\n' + '=' * 60)
    print('MRT-UMamba 训练')
    print(f'  - 训练轮数: {start_epoch} -> {CONFIG["total_epochs"]}')
    print(f'  - SAM rho: {CONFIG["rho"]}')
    print('提示: 按 Ctrl+C 可安全暂停，下次运行自动继续')
    print('=' * 60)

    try:
        for epoch in range(start_epoch, CONFIG['total_epochs'] + 1):
            print(f'\nEpoch {epoch}/{CONFIG["total_epochs"]}')
            model.train()
            train_loss_sum = 0.0
            pbar = tqdm(train_loader, desc='Training')
            stage2 = False

            for imgs, masks in pbar:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                boundary_targets = []
                skeleton_targets = []
                for i in range(masks.size(0)):
                    mask_np = masks[i].detach().cpu().numpy()
                    boundary_targets.append(boundary_builder(mask_np))
                    skeleton_targets.append(skeleton_builder(mask_np))
                boundary_targets = torch.from_numpy(np.stack(boundary_targets, axis=0)).unsqueeze(1).float().to(device)
                skeleton_targets = torch.from_numpy(np.stack(skeleton_targets, axis=0)).unsqueeze(1).float().to(device)

                pos_weight = compute_pos_weight(masks)
                seg_loss_fn.pos_weight = pos_weight
                boundary_pos_weight = compute_pos_weight(boundary_targets)
                skeleton_pos_weight = compute_pos_weight(skeleton_targets)

                refined_logits, boundary_logits, skeleton_logits, coarse_logits, uncertainty, uncertainty_gate = model(imgs, return_aux=True)

                coarse_loss = seg_loss_fn(coarse_logits, masks)
                refined_loss = seg_loss_fn(refined_logits, masks)
                boundary_loss = F.binary_cross_entropy_with_logits(
                    boundary_logits,
                    boundary_targets,
                    pos_weight=boundary_pos_weight.to(device),
                )
                skeleton_loss = F.binary_cross_entropy_with_logits(
                    skeleton_logits,
                    skeleton_targets,
                    pos_weight=skeleton_pos_weight.to(device),
                )
                uncertain_loss = topk_uncertain_loss(refined_logits, masks, uncertainty, ratio=0.25)
                background = 1.0 - masks.float()
                refined_prob = torch.sigmoid(refined_logits[:, 1]) if refined_logits.shape[1] > 1 else torch.sigmoid(refined_logits[:, 0])
                coarse_prob = torch.sigmoid(coarse_logits[:, 1]).detach() if coarse_logits.shape[1] > 1 else torch.sigmoid(coarse_logits[:, 0]).detach()
                consistency_loss = F.l1_loss(refined_prob * background, coarse_prob * background)
                gate_penalty = torch.mean(uncertainty_gate * background.unsqueeze(1))
                bridge_target = torch.clamp(F.max_pool2d(skeleton_targets, kernel_size=3, stride=1, padding=1), 0.0, 1.0)
                bridge_loss = F.binary_cross_entropy(refined_prob.unsqueeze(1), bridge_target)

                if stage2:
                    loss = (
                        CONFIG['coarse_weight'] * coarse_loss +
                        CONFIG['fine_weight'] * refined_loss +
                        CONFIG['boundary_weight'] * boundary_loss +
                        CONFIG['skeleton_weight'] * skeleton_loss +
                        CONFIG['uncertain_weight'] * uncertain_loss +
                        CONFIG['consistency_weight'] * consistency_loss +
                        CONFIG['bridge_weight'] * bridge_loss +
                        0.03 * gate_penalty
                    )
                else:
                    loss = (
                        CONFIG['coarse_weight'] * coarse_loss +
                        CONFIG['fine_weight'] * refined_loss +
                        0.5 * CONFIG['consistency_weight'] * consistency_loss +
                        0.5 * CONFIG['bridge_weight'] * bridge_loss
                    )

                loss.backward()
                optimizer.first_step(zero_grad=True)

                refined_logits2, boundary_logits2, skeleton_logits2, coarse_logits2, uncertainty2, uncertainty_gate2 = model(imgs, return_aux=True)
                coarse_loss2 = seg_loss_fn(coarse_logits2, masks)
                refined_loss2 = seg_loss_fn(refined_logits2, masks)
                boundary_loss2 = F.binary_cross_entropy_with_logits(
                    boundary_logits2,
                    boundary_targets,
                    pos_weight=boundary_pos_weight.to(device),
                )
                skeleton_loss2 = F.binary_cross_entropy_with_logits(
                    skeleton_logits2,
                    skeleton_targets,
                    pos_weight=skeleton_pos_weight.to(device),
                )
                uncertain_loss2 = topk_uncertain_loss(refined_logits2, masks, uncertainty2, ratio=0.25)
                refined_prob2 = torch.sigmoid(refined_logits2[:, 1]) if refined_logits2.shape[1] > 1 else torch.sigmoid(refined_logits2[:, 0])
                coarse_prob2 = torch.sigmoid(coarse_logits2[:, 1]).detach() if coarse_logits2.shape[1] > 1 else torch.sigmoid(coarse_logits2[:, 0]).detach()
                consistency_loss2 = F.l1_loss(refined_prob2 * background, coarse_prob2 * background)
                gate_penalty2 = torch.mean(uncertainty_gate2 * background.unsqueeze(1))
                bridge_target2 = torch.clamp(F.max_pool2d(skeleton_targets, kernel_size=3, stride=1, padding=1), 0.0, 1.0)
                bridge_loss2 = F.binary_cross_entropy(refined_prob2.unsqueeze(1), bridge_target2)

                if stage2:
                    loss2 = (
                        CONFIG['coarse_weight'] * coarse_loss2 +
                        CONFIG['fine_weight'] * refined_loss2 +
                        CONFIG['boundary_weight'] * boundary_loss2 +
                        CONFIG['skeleton_weight'] * skeleton_loss2 +
                        CONFIG['uncertain_weight'] * uncertain_loss2 +
                        CONFIG['consistency_weight'] * consistency_loss2 +
                        CONFIG['bridge_weight'] * bridge_loss2 +
                        0.03 * gate_penalty2
                    )
                else:
                    loss2 = (
                        CONFIG['coarse_weight'] * coarse_loss2 +
                        CONFIG['fine_weight'] * refined_loss2 +
                        0.5 * CONFIG['consistency_weight'] * consistency_loss2 +
                        0.5 * CONFIG['bridge_weight'] * bridge_loss2
                    )

                loss2.backward()
                optimizer.second_step(zero_grad=True)
                update_ema_model(ema_model, model, decay=0.995)

                train_loss_sum += loss.item()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'coarse': f'{coarse_loss.item():.4f}',
                    'ref': f'{refined_loss.item():.4f}'
                })

            scheduler.step()
            best_val = evaluate_thresholds(ema_model, val_loader, device)
            avg_train_loss = train_loss_sum / max(1, len(train_loader))

            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Stage: {"fine-tune" if stage2 else "warmup"}')
            print(f'Val Precision: {best_val["precision"]:.4f}')
            print(f'Val Recall: {best_val["recall"]:.4f}')
            print(f'Val F1: {best_val["f1"]:.4f}')
            print(f'Val Best Threshold: {best_val["threshold"]:.2f}')

            save_checkpoint(
                checkpoint_file,
                epoch,
                model,
                optimizer,
                max(best_f1, best_val['f1']),
                best_threshold if best_f1 >= best_val['f1'] else best_val['threshold'],
            )

            if best_val['f1'] > best_f1:
                best_f1 = best_val['f1']
                best_threshold = best_val['threshold']
                torch.save(ema_model.state_dict(), best_model_file)
                save_checkpoint(
                    best_state_file,
                    epoch,
                    ema_model,
                    optimizer,
                    best_f1,
                    best_threshold,
                    extra={'metrics': best_val},
                )
                print(f'✓ 保存最佳模型 (F1={best_f1:.4f}, thr={best_threshold:.2f})')

            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], f'model_epoch{epoch}.pth'))

    except KeyboardInterrupt:
        print('\n\n' + '=' * 50)
        print('用户中断训练')
        save_checkpoint(checkpoint_file, epoch, model, optimizer, best_f1, best_threshold)
        print(f'当前进度已保存到: {checkpoint_file}')
        print('下次运行可继续训练')
        print('=' * 50)
        return

    print(f'\n训练完成！最佳验证 F1: {best_f1:.4f}, 最佳阈值: {best_threshold:.2f}')


if __name__ == '__main__':
    train()