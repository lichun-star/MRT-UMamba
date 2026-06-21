import argparse
import importlib.util
import os
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_ROOT = "/root/autodl-tmp/Demo2/deepcrack"
DEFAULT_DATA_PATH = os.path.join(DEFAULT_ROOT, "CRACK500")
DEFAULT_MODEL_FILE = os.path.join(DEFAULT_ROOT, "crack500.py")
DEFAULT_CHECKPOINT = os.path.join(DEFAULT_ROOT, "checkpoints_crackaware_skeleton_scan", "best_model.pth")


class Crack500Dataset(Dataset):
    def __init__(self, data_path: str, split: str = "test", img_size: int = 512):
        self.img_size = img_size
        self.split = split

        if split == "train":
            self.img_dir = os.path.join(data_path, "train_images_512")
            self.mask_dir = os.path.join(data_path, "train_labels_512")
        else:
            self.img_dir = os.path.join(data_path, "test_images_512")
            self.mask_dir = os.path.join(data_path, "test_labels_512")

        import glob

        self.img_files = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg")) + glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        print(f"{split}: found {len(self.img_files)} images")

    def _mask_path_from_img(self, img_path: str) -> str:
        base = os.path.basename(img_path)
        name = os.path.splitext(base)[0]
        return os.path.join(self.mask_dir, f"{name}.png")

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        mask_path = self._mask_path_from_img(img_path)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = (mask > 128).astype(np.float32)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()


def load_model_module(model_file: str):
    spec = importlib.util.spec_from_file_location("crack_model_module", model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model module from: {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strip_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "ema_model.", "model.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "ema_state_dict", "ema_model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint  # type: ignore[return-value]
    raise RuntimeError("Unsupported checkpoint format")


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = strip_prefix(extract_state_dict(checkpoint))

    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} tensors from checkpoint")
    if skipped:
        print(f"Skipped {len(skipped)} tensors with unmatched shapes/keys")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")


def build_model(model_module, device: torch.device):
    model_cls = getattr(model_module, "CrackAwareSkeletonScanUMamba", None)
    if model_cls is None:
        raise AttributeError("Model file must define CrackAwareSkeletonScanUMamba")

    model = model_cls()
    model.to(device)
    return model


@torch.no_grad()
def predict_logits(model: torch.nn.Module, image: torch.Tensor, tta: str, scales: List[float]) -> torch.Tensor:
    logits_sum: Optional[torch.Tensor] = None
    count = 0

    def forward_once(inp: torch.Tensor) -> torch.Tensor:
        output = model(inp)
        if isinstance(output, (list, tuple)):
            output = output[0]
        return output

    if tta == "none":
        output = forward_once(image)
        return F.interpolate(output, size=image.shape[-2:], mode="bilinear", align_corners=False)

    def add_logits(inp: torch.Tensor) -> None:
        nonlocal logits_sum, count
        output = forward_once(inp)
        output = F.interpolate(output, size=image.shape[-2:], mode="bilinear", align_corners=False)
        if logits_sum is None:
            logits_sum = output
        else:
            logits_sum = logits_sum + output
        count += 1

    def forward_flips(inp: torch.Tensor) -> None:
        nonlocal logits_sum, count
        add_logits(inp)
        flipped_h = torch.flip(inp, dims=[3])
        output_h = forward_once(flipped_h)
        output_h = torch.flip(output_h, dims=[3])
        output_h = F.interpolate(output_h, size=image.shape[-2:], mode="bilinear", align_corners=False)
        if logits_sum is None:
            logits_sum = output_h
        else:
            logits_sum = logits_sum + output_h
        count += 1

    if tta == "flip":
        forward_flips(image)
    elif tta in {"ms", "ms_flip"}:
        for scale in scales:
            if abs(scale - 1.0) < 1e-6:
                scaled = image
            else:
                scaled = F.interpolate(image, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)

            if tta == "ms":
                add_logits(scaled)
            else:
                forward_flips(scaled)
    else:
        raise ValueError(f"Unknown tta mode: {tta}")

    assert logits_sum is not None and count > 0
    return logits_sum / count


def compute_binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    计算二分类分割指标，包括 IoU（裂纹类别）和 mIoU（所有类别平均）
    """
    pred = pred.float().reshape(-1)
    target = target.float().reshape(-1)

    tp = torch.sum(pred * target).item()
    fp = torch.sum(pred * (1.0 - target)).item()
    fn = torch.sum((1.0 - pred) * target).item()
    tn = torch.sum((1.0 - pred) * (1.0 - target)).item()

    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)

    # IoU: 裂纹类别
    iou = tp / (tp + fp + fn + eps)

    # IoU: 背景类别
    iou_bg = tn / (tn + fp + fn + eps)

    # mIoU: 所有类别平均
    miou = (iou + iou_bg) / 2.0

    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "iou_bg": iou_bg,
        "miou": miou,
    }


@torch.no_grad()
def evaluate_thresholds(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: Iterable[float],
    tta: str,
    scales: List[float],
) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {
        "precision": 0.0,
        "recall": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "iou_bg": 0.0,
        "miou": 0.0,
    }
    best_dice = -1.0

    threshold_list = list(thresholds)
    for threshold in threshold_list:
        total_tp = 0.0
        total_fp = 0.0
        total_fn = 0.0
        total_tn = 0.0

        for images, masks in tqdm(loader, desc=f"threshold={threshold:.2f}", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            logits = predict_logits(model, images, tta=tta, scales=scales)
            if logits.shape[1] > 1:
                probs = torch.softmax(logits, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(logits[:, 0])

            preds = (probs > threshold).float()

            total_tp += torch.sum(preds * masks).item()
            total_fp += torch.sum(preds * (1.0 - masks)).item()
            total_fn += torch.sum((1.0 - preds) * masks).item()
            total_tn += torch.sum((1.0 - preds) * (1.0 - masks)).item()

        eps = 1e-8

        precision = total_tp / (total_tp + total_fp + eps)
        recall = total_tp / (total_tp + total_fn + eps)
        dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + eps)

        # IoU: 裂纹类别
        iou = total_tp / (total_tp + total_fp + total_fn + eps)

        # IoU: 背景类别
        iou_bg = total_tn / (total_tn + total_fp + total_fn + eps)

        # mIoU: 所有类别平均
        miou = (iou + iou_bg) / 2.0

        # 打印每个阈值的详细结果
        print(f"  thr={threshold:.2f}: F1={dice*100:.2f}%, IoU={iou*100:.2f}%, mIoU={miou*100:.2f}%")

        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "dice": dice,
                "iou": iou,
                "iou_bg": iou_bg,
                "miou": miou,
            }

    return best_threshold, best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate crack-aware scan model on Crack500 test split")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold-min", type=float, default=0.30)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--threshold-steps", type=int, default=9)
    parser.add_argument("--tta", choices=["none", "flip", "ms", "ms_flip"], default="ms_flip")
    parser.add_argument("--scales", type=float, nargs="*", default=[0.75, 1.0, 1.25])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"Model file not found: {args.model_file}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    model_module = load_model_module(args.model_file)
    model = build_model(model_module, device)
    load_checkpoint_into_model(model, args.checkpoint)
    model.eval()

    dataset = Crack500Dataset(args.data_path, split="test", img_size=args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps).tolist()
    best_threshold, best_metrics = evaluate_thresholds(
        model=model,
        loader=loader,
        device=device,
        thresholds=thresholds,
        tta=args.tta,
        scales=list(args.scales),
    )

    print("\n" + "="*60)
    print("=== Test Result (Best threshold based on F1) ===")
    print("="*60)
    print(f"Best threshold:     {best_threshold:.3f}")
    print(f"Dice / F1:          {best_metrics['dice'] * 100:.2f}%")
    print(f"IoU (crack class):  {best_metrics['iou'] * 100:.2f}%")
    print(f"IoU (background):   {best_metrics['iou_bg'] * 100:.2f}%")
    print(f"mIoU (all classes): {best_metrics['miou'] * 100:.2f}%")
    print(f"Precision:          {best_metrics['precision'] * 100:.2f}%")
    print(f"Recall:             {best_metrics['recall'] * 100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()