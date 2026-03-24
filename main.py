#!/usr/bin/env python3
"""
PARSeq Training Pipeline with Legibility Classification
========================================================

This script trains a PARSeq model on tracklet images while computing legibility scores.

Usage:
    python main.py                          # Train with default settings (1/5 data, 10 epochs)
    python main.py --full-data             # Train with all data
    python main.py --epochs 20             # Train for 20 epochs instead
    python main.py --data-fraction 0.5     # Use 50% of data
"""

import os
import sys
import json
import argparse
import subprocess
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
from enum import Enum
import logging
import random

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import tqdm

# Improve Tensor Core utilization on modern NVIDIA GPUs.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

try:
    import lmdb
except ImportError:
    lmdb = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# LEGIBILITY CLASSIFIER (Extracted from cosc_419_project.py)
# ============================================================================

class LegibilityStatus(Enum):
    """Legibility classification status"""
    LEGIBLE = "legible"
    UNKNOWN = "unknown"
    BLURRY = "blurry"
    DARK = "dark"
    BRIGHT = "bright"
    OCCLUDED = "occluded"
    SMALL = "small"


class LegibilityClassifier34(nn.Module):
    """ResNet34-based legibility classifier"""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
    
    def forward(self, x):
        return self.backbone(x)


class HeuristicLegibilityClassifier:
    """
    Fast heuristic-based legibility classifier using image quality metrics
    (Laplacian sharpness, brightness, contrast, detection size)
    """
    
    BLUR_THRESHOLD = 25.0
    BRIGHTNESS_MIN = 30
    BRIGHTNESS_MAX = 225
    CONTRAST_MIN = 20
    JERSEY_SIZE_MIN = 20
    
    def __init__(self):
        pass
    
    def classify(self, img_path: str) -> Dict:
        """Classify image legibility using heuristics"""
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {
                    'image': img_path,
                    'legible': False,
                    'status': LegibilityStatus.UNKNOWN.value,
                    'details': 'Could not read image'
                }
            
            # Compute metrics
            sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
            brightness = np.mean(img)
            contrast = np.std(img)
            
            # Estimate jersey/region size
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = max([cv2.contourArea(c) for c in contours]) if contours else 0
            jersey_size = np.sqrt(max_area)
            
            # Heuristic decision
            status, legible = self._heuristic_decision(
                sharpness, brightness, contrast, jersey_size, 1.0
            )
            
            return {
                'image': img_path,
                'legible': legible,
                'status': status.value,
                'metrics': {
                    'sharpness': float(sharpness),
                    'brightness': float(brightness),
                    'contrast': float(contrast),
                    'jersey_size': float(jersey_size)
                }
            }
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
            return {
                'image': img_path,
                'legible': False,
                'status': LegibilityStatus.UNKNOWN.value,
                'details': str(e)
            }
    
    @staticmethod
    def _heuristic_decision(sharpness: float, brightness: float, contrast: float,
                           jersey_size: float, score: float) -> Tuple[LegibilityStatus, bool]:
        """Heuristic decision logic"""
        if sharpness < HeuristicLegibilityClassifier.BLUR_THRESHOLD:
            return LegibilityStatus.BLURRY, False
        if brightness < HeuristicLegibilityClassifier.BRIGHTNESS_MIN:
            return LegibilityStatus.DARK, False
        if brightness > HeuristicLegibilityClassifier.BRIGHTNESS_MAX:
            return LegibilityStatus.BRIGHT, False
        if contrast < HeuristicLegibilityClassifier.CONTRAST_MIN:
            return LegibilityStatus.OCCLUDED, False
        if jersey_size < HeuristicLegibilityClassifier.JERSEY_SIZE_MIN:
            return LegibilityStatus.SMALL, False
        
        return (LegibilityStatus.LEGIBLE if score > 0.5 else LegibilityStatus.UNKNOWN), bool(score > 0.5)


# ============================================================================
# DATA PREPARATION & LEGIBILITY SCORING
# ============================================================================

def load_ground_truth(gt_path: Path) -> Dict[int, int]:
    """
    Load ground truth labels from train_gt.json
    
    Returns:
        Dictionary mapping tracklet_id -> jersey_number (-1 for invalid)
    """
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    with open(gt_path, 'r') as f:
        data = json.load(f)
    
    # Convert string keys to int
    gt = {int(k): v for k, v in data.items()}
    logger.info(f"Loaded ground truth for {len(gt)} tracklets")
    return gt


def discover_tracklets(image_dir: Path, ground_truth: Optional[Dict[int, int]] = None) -> List[Tuple[Path, Optional[str]]]:
    """
    Discover all tracklet directories in train/image/
    
    Expected structure:
        train/image/
            0/
                img_1.jpg
                img_2.jpg
            1/
                img_3.jpg
    
    Returns:
        List of tuples: (tracklet_path, label) where label is jersey number or None if invalid
    """
    tracklets = []
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    for tracklet_dir in sorted(image_dir.iterdir()):
        if tracklet_dir.is_dir() and tracklet_dir.name.isdigit():
            tracklet_id = int(tracklet_dir.name)
            
            # Get label from ground truth if available
            label = None
            if ground_truth is not None:
                jersey_num = ground_truth.get(tracklet_id, -1)
                if jersey_num >= 0:  # Skip invalid labels (-1)
                    label = str(jersey_num)
            
            tracklets.append((tracklet_dir, label))
    
    valid_count = sum(1 for _, label in tracklets if label is not None)
    logger.info(f"Discovered {len(tracklets)} tracklets ({valid_count} with valid labels)")
    return tracklets


def sample_tracklets(tracklets: List[Tuple[Path, Optional[str]]], fraction: float) -> List[Tuple[Path, Optional[str]]]:
    """Sample a fraction of tracklets (only from valid ones)"""
    # Keep only tracklets with valid labels
    valid_tracklets = [t for t in tracklets if t[1] is not None]
    
    n_samples = max(1, int(len(valid_tracklets) * fraction))
    sampled = valid_tracklets[:n_samples]
    logger.info(f"Using {len(sampled)}/{len(valid_tracklets)} valid tracklets ({fraction*100:.0f}%)")
    return sampled


def collect_samples(
    tracklets: List[Tuple[Path, Optional[str]]],
    legibility_classifier: Optional[HeuristicLegibilityClassifier] = None,
    filter_by_legibility: bool = False,
) -> Tuple[List[Tuple[Path, str]], Dict, Dict[str, int]]:
    """Flatten tracklet folders into (image_path, label) pairs and report filter stats."""
    samples: List[Tuple[Path, str]] = []
    legibility_data: Dict = {}
    total_images = 0
    flagged_illegible = 0
    filtered_by_legibility = 0

    for tracklet_path, label in tqdm.tqdm(tracklets, desc="Collecting samples"):
        if label is None:
            continue
        images = sorted([p for p in tracklet_path.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        for img_path in images:
            total_images += 1
            if legibility_classifier:
                result = legibility_classifier.classify(str(img_path))
                legibility_data[str(img_path)] = result
                if not result.get('legible', False):
                    flagged_illegible += 1
                    if filter_by_legibility:
                        filtered_by_legibility += 1
                        continue
            samples.append((img_path, label))

    stats = {
        'total_images_seen': total_images,
        'images_kept_for_training': len(samples),
        'images_flagged_illegible': flagged_illegible,
        'images_filtered_by_legibility': filtered_by_legibility,
    }

    logger.info(
        "Collected %d labeled images from %d tracklets | kept=%d flagged_illegible=%d filtered_by_legibility=%d",
        total_images,
        len(tracklets),
        stats['images_kept_for_training'],
        stats['images_flagged_illegible'],
        stats['images_filtered_by_legibility'],
    )
    return samples, legibility_data, stats


def split_samples(
    samples: List[Tuple[Path, str]], val_fraction: float = 0.1, seed: int = 42
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    """Create train/val split at image level for PARSeq training."""
    if not samples:
        raise ValueError("No labeled samples found. Check tracklets and train_gt.json.")
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    n_val = min(n_val, len(shuffled) - 1)
    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    logger.info(f"Split samples -> train: {len(train_samples)}, val: {len(val_samples)}")
    return train_samples, val_samples


def split_samples_by_tracklet(
    samples: List[Tuple[Path, str]], val_fraction: float = 0.1, seed: int = 42
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    """Create train/val split at tracklet level to avoid frame leakage across splits."""
    if not samples:
        raise ValueError("No labeled samples found. Check tracklets and train_gt.json.")

    by_tracklet: Dict[str, List[Tuple[Path, str]]] = {}
    for img_path, label in samples:
        tracklet_id = img_path.parent.name
        by_tracklet.setdefault(tracklet_id, []).append((img_path, label))

    tracklet_ids = list(by_tracklet.keys())
    if len(tracklet_ids) < 2:
        raise ValueError("Need at least 2 tracklets to build train/val split without leakage.")

    rng = random.Random(seed)
    rng.shuffle(tracklet_ids)

    n_val_tracklets = max(1, int(len(tracklet_ids) * val_fraction))
    n_val_tracklets = min(n_val_tracklets, len(tracklet_ids) - 1)
    val_tracklets = set(tracklet_ids[:n_val_tracklets])

    train_samples: List[Tuple[Path, str]] = []
    val_samples: List[Tuple[Path, str]] = []
    for tid, group in by_tracklet.items():
        if tid in val_tracklets:
            val_samples.extend(group)
        else:
            train_samples.extend(group)

    logger.info(
        "Split by tracklet -> train: %d images (%d tracklets), val: %d images (%d tracklets)",
        len(train_samples),
        len(tracklet_ids) - len(val_tracklets),
        len(val_samples),
        len(val_tracklets),
    )
    return train_samples, val_samples


def load_precomputed_kept_samples(kept_path: Path) -> List[Tuple[Path, str]]:
    """Load pre-scored kept samples from legibility_kept_samples.txt (tab-separated path\tlabel)."""
    if not kept_path.exists():
        raise FileNotFoundError(f"Precomputed kept samples file not found: {kept_path}")
    samples = []
    with open(kept_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                logger.warning(f"Skipping malformed line: {line!r}")
                continue
            img_path, label = Path(parts[0]), parts[1].strip()
            if img_path.exists():
                samples.append((img_path, label))
            else:
                logger.warning(f"Image not found, skipping: {img_path}")
    logger.info(f"Loaded {len(samples)} pre-scored kept samples from {kept_path}")
    return samples


def _read_image_bytes(image_path: Path) -> bytes:
    """Ensure image bytes are RGB-readable, converting non-JPEG inputs when needed."""
    if image_path.suffix.lower() in ['.jpg', '.jpeg']:
        return image_path.read_bytes()

    img = Image.open(image_path).convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def write_parseq_lmdb(samples: List[Tuple[Path, str]], lmdb_dir: Path) -> None:
    """Write LMDB with PARSeq/STRHub expected keys: image-*, label-*, num-samples."""
    if lmdb is None:
        raise RuntimeError("lmdb is not installed. Install with: pip install lmdb")

    lmdb_dir.mkdir(parents=True, exist_ok=True)

    # Use a practical map size estimate to avoid huge preallocation failures on Windows.
    estimated_bytes = 0
    for img_path, label in samples:
        try:
            estimated_bytes += img_path.stat().st_size
        except OSError:
            estimated_bytes += 200_000  # conservative fallback per image
        estimated_bytes += len(label) + 64  # label + LMDB key/value overhead

    min_map = 512 * 1024 * 1024          # 512 MB
    max_map = 32 * 1024 * 1024 * 1024    # 32 GB safety cap
    map_size = int(estimated_bytes * 1.4)  # headroom for metadata/overhead
    map_size = max(min_map, min(map_size, max_map))
    logger.info(f"Opening LMDB at {lmdb_dir} with map_size={map_size / (1024**3):.2f} GB")

    env = lmdb.open(str(lmdb_dir), map_size=map_size)
    cache = {}
    count = 1

    for img_path, label in tqdm.tqdm(samples, desc=f"Writing {lmdb_dir.name}"):
        try:
            image_bin = _read_image_bytes(img_path)
            cache[f'image-{count:09d}'.encode()] = image_bin
            cache[f'label-{count:09d}'.encode()] = label.encode('utf-8')
            if count % 1000 == 0:
                with env.begin(write=True) as txn:
                    for k, v in cache.items():
                        txn.put(k, v)
                cache = {}
            count += 1
        except Exception as e:
            logger.warning(f"Skipping {img_path}: {e}")

    cache[b'num-samples'] = str(count - 1).encode()
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)
    env.close()
    logger.info(f"Created LMDB at {lmdb_dir} with {count - 1} samples")


def create_parseq_dataset(
    tracklets: List[Tuple[Path, Optional[str]]],
    dataset_root: Path,
    legibility_classifier: Optional[HeuristicLegibilityClassifier],
    val_fraction: float,
    seed: int,
    filter_by_legibility: bool,
) -> Tuple[Dict, Dict[str, int]]:
    """
    Create PARSeq-compatible dataset tree:
      dataset_root/train/custom/data.mdb
      dataset_root/val/custom/data.mdb
    """
    samples, legibility_data, stats = collect_samples(
        tracklets,
        legibility_classifier,
        filter_by_legibility=filter_by_legibility,
    )
    train_samples, val_samples = split_samples_by_tracklet(samples, val_fraction=val_fraction, seed=seed)

    train_lmdb_dir = dataset_root / 'train' / 'custom'
    val_lmdb_dir = dataset_root / 'val' / 'custom'

    # Remove stale partial LMDBs from previous failed runs.
    if train_lmdb_dir.exists():
        shutil.rmtree(train_lmdb_dir)
    if val_lmdb_dir.exists():
        shutil.rmtree(val_lmdb_dir)

    write_parseq_lmdb(train_samples, train_lmdb_dir)
    write_parseq_lmdb(val_samples, val_lmdb_dir)

    if legibility_data:
        leg_path = dataset_root / 'legibility_scores.json'
        with open(leg_path, 'w', encoding='utf-8') as f:
            json.dump(legibility_data, f, indent=2)
        logger.info(f"Saved legibility scores to {leg_path}")

    return legibility_data, stats


# ============================================================================
# TRAINING
# ============================================================================

def train_parseq(dataset_root: Path, epochs: int, output_dir: Path, pretrained_model: Optional[Path] = None, max_label_length: int = 2) -> str:
    """
    Train PARSeq model using Hydra configuration.
    
    Args:
        dataset_root: Path to the LMDB dataset root directory
        epochs: Number of training epochs
        output_dir: Output directory for checkpoints and logs
        pretrained_model: Optional path to pretrained checkpoint to fine-tune from
    
    Returns:
        Path to best checkpoint
    """
    parseq_dir = Path("parseq-main").resolve()
    output_dir = output_dir.resolve()
    run_dir = output_dir / "train_run"

    cmd = [
        sys.executable, "train.py",
        f"trainer.max_epochs={epochs}",
        f"trainer.accelerator={'gpu' if torch.cuda.is_available() else 'cpu'}",
        f"trainer.devices=1",
        "trainer.val_check_interval=1.0",
        "+trainer.check_val_every_n_epoch=1",
        f"data.root_dir={str(dataset_root.resolve())}",
        "data.train_dir=custom",
        "charset=10_digits",
        f"model.max_label_length={max_label_length}",
        f"data.max_label_length={max_label_length}",
        f"model.batch_size=128",
        f"hydra.run.dir={str(run_dir)}",
    ]
    
    # Add pretrained model if specified
    if pretrained_model is not None:
        pretrained_full_path = (
            Path.cwd() / pretrained_model
            if not pretrained_model.is_absolute()
            else pretrained_model
        )
        # Use forward slashes so Hydra/OmegaConf doesn't choke on Windows backslashes.
        # Wrap in escaped double-quotes so the '=' chars in checkpoint names don't
        # confuse the Hydra override parser (key=value splitting).
        safe_path = pretrained_full_path.as_posix()
        cmd.append(f'pretrained="{safe_path}"')

    logger.info(f"Training interpreter: {sys.executable}")
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(parseq_dir), capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    # Find checkpoint
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        checkpoints = list(ckpt_dir.glob("*.ckpt"))
        if checkpoints:
            # Prefer metric checkpoints (epoch=...-val_accuracy=...). Fall back to last.ckpt.
            metric_ckpts = [c for c in checkpoints if c.name.startswith("epoch=") and "val_accuracy=" in c.name]
            if metric_ckpts:
                def _val_acc(path: Path) -> float:
                    try:
                        segment = path.name.split("val_accuracy=")[1].split("-")[0]
                        return float(segment)
                    except Exception:
                        return float("-inf")

                best_ckpt = max(metric_ckpts, key=_val_acc)
                logger.info(f"Selected best checkpoint by val_accuracy: {best_ckpt.name}")
                return str(best_ckpt)

            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists():
                logger.info("Selected fallback checkpoint: last.ckpt")
                return str(last_ckpt)

            return str(sorted(checkpoints)[-1])
    
    raise FileNotFoundError("No checkpoint found after training")


def main():
    parser = argparse.ArgumentParser(description="PARSeq Training Pipeline with Legibility Classification")
    parser.add_argument("--data-fraction", type=float, default=0.2,
                       help="Fraction of data to use (default: 0.2 = 1/5)")
    parser.add_argument("--full-data", action="store_true",
                       help="Use all available data (overrides --data-fraction)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs (default: 10)")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/parseq_train"),
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--skip-legibility", action="store_true",
                       help="Skip legibility classification")
    parser.add_argument("--legibility-precomputed", type=Path, default=None,
                       help="Path to pre-scored legibility_kept_samples.txt; skips scoring and uses these images directly")
    parser.add_argument("--filter-by-legibility", action="store_true",
                       help="Actually remove images predicted as illegible (default: keep all and only report stats)")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                       help="Fraction of tracklets for validation split (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for train/val split")
    parser.add_argument("--gt-path", type=Path, default=Path("./train/train_gt.json"),
                       help="Path to ground truth JSON file")
    parser.add_argument("--pretrained-model", type=Path, default=None,
                       help="Path to pretrained PARSeq checkpoint to fine-tune from")
    parser.add_argument("--max-label-length", type=int, default=2,
                       help="Maximum target label length (default: 2 for jersey numbers)")
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path.cwd()
    image_dir = project_root / "train" / "image"
    gt_path = project_root / args.gt_path if not args.gt_path.is_absolute() else args.gt_path
    data_fraction = 1.0 if args.full_data else args.data_fraction
    
    logger.info("=" * 80)
    logger.info("PARSeq Training Pipeline with Ground Truth Labels")
    logger.info("=" * 80)
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"Ground truth: {gt_path}")
    logger.info(f"Data fraction: {data_fraction*100:.0f}%")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Filter by legibility: {args.filter_by_legibility}")
    logger.info(f"Legibility precomputed file: {args.legibility_precomputed}")

    dataset_root = args.output_dir / "parseq_data"

    if args.legibility_precomputed is not None:
        # ------------------------------------------------------------------ #
        # Fast path: use pre-scored kept samples, skip GT loading + scoring   #
        # ------------------------------------------------------------------ #
        logger.info("\n[Step 1] Loading pre-scored kept samples (skipping GT / legibility scoring)...")
        precomputed_path = (
            Path.cwd() / args.legibility_precomputed
            if not args.legibility_precomputed.is_absolute()
            else args.legibility_precomputed
        )
        samples = load_precomputed_kept_samples(precomputed_path)

        logger.info("\n[Step 2] Building train/val split from pre-scored samples...")
        train_samples, val_samples = split_samples_by_tracklet(samples, val_fraction=args.val_fraction, seed=args.seed)

        logger.info("\n[Step 3] Writing PARSeq LMDB dataset...")
        train_lmdb_dir = dataset_root / 'train' / 'custom'
        val_lmdb_dir = dataset_root / 'val' / 'custom'
        if train_lmdb_dir.exists():
            shutil.rmtree(train_lmdb_dir)
        if val_lmdb_dir.exists():
            shutil.rmtree(val_lmdb_dir)
        write_parseq_lmdb(train_samples, train_lmdb_dir)
        write_parseq_lmdb(val_samples, val_lmdb_dir)
        logger.info(f"Dataset ready: {len(train_samples)} train, {len(val_samples)} val")

    else:
        # ------------------------------------------------------------------ #
        # Normal path: discover tracklets, optionally score + filter          #
        # ------------------------------------------------------------------ #
        # Step 0: Load ground truth
        logger.info("\n[Step 0] Loading ground truth...")
        ground_truth = load_ground_truth(gt_path)

        # Step 1: Discover tracklets
        logger.info("\n[Step 1] Discovering tracklets...")
        tracklets = discover_tracklets(image_dir, ground_truth)
        tracklets = sample_tracklets(tracklets, data_fraction)

        # Step 2: Compute legibility scores
        logger.info("\n[Step 2] Computing legibility scores...")
        legibility_classifier = None if args.skip_legibility else HeuristicLegibilityClassifier()

        # Step 3: Create dataset
        logger.info("\n[Step 3] Creating PARSeq LMDB dataset...")
        legibility_data, data_stats = create_parseq_dataset(
            tracklets,
            dataset_root=dataset_root,
            legibility_classifier=legibility_classifier,
            val_fraction=args.val_fraction,
            seed=args.seed,
            filter_by_legibility=args.filter_by_legibility,
        )
        logger.info(
            "Legibility summary: total=%d kept=%d flagged_illegible=%d filtered=%d",
            data_stats['total_images_seen'],
            data_stats['images_kept_for_training'],
            data_stats['images_flagged_illegible'],
            data_stats['images_filtered_by_legibility'],
        )
    
    # Step 4: Train PARSeq
    logger.info("\n[Step 4] Training PARSeq model...")
    try:
        checkpoint = train_parseq(
            dataset_root,
            args.epochs,
            args.output_dir,
            args.pretrained_model,
            args.max_label_length,
        )
        logger.info(f"\n✅ Training completed!")
        logger.info(f"   Checkpoint: {checkpoint}")
        logger.info(f"   Legibility scores: {dataset_root / 'legibility_scores.json'}")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
