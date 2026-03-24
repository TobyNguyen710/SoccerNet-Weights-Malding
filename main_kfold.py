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


def create_kfold_splits(
    samples: List[Tuple[Path, str]], k_folds: int = 5, seed: int = 42
) -> List[Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]]:
    """Create K-fold train/val splits at image level for PARSeq training."""
    if not samples:
        raise ValueError("No labeled samples found. Check tracklets and train_gt.json.")
    if k_folds < 2:
        raise ValueError("k_folds must be at least 2.")
    if len(samples) < k_folds:
        raise ValueError(f"Not enough samples ({len(samples)}) for {k_folds} folds.")

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    fold_sizes = [len(shuffled) // k_folds] * k_folds
    for i in range(len(shuffled) % k_folds):
        fold_sizes[i] += 1

    folds: List[List[Tuple[Path, str]]] = []
    start = 0
    for fold_size in fold_sizes:
        end = start + fold_size
        folds.append(shuffled[start:end])
        start = end

    split_pairs: List[Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]] = []
    for fold_idx in range(k_folds):
        val_samples = folds[fold_idx]
        train_samples: List[Tuple[Path, str]] = []
        for i in range(k_folds):
            if i != fold_idx:
                train_samples.extend(folds[i])
        split_pairs.append((train_samples, val_samples))
        logger.info(
            "Fold %d/%d split -> train: %d, val: %d",
            fold_idx + 1,
            k_folds,
            len(train_samples),
            len(val_samples),
        )

    return split_pairs


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


def write_parseq_dataset_split(
    train_samples: List[Tuple[Path, str]],
    val_samples: List[Tuple[Path, str]],
    dataset_root: Path,
) -> None:
    """Create PARSeq-compatible LMDB tree for a single fold split."""
    train_lmdb_dir = dataset_root / 'train' / 'custom'
    val_lmdb_dir = dataset_root / 'val' / 'custom'

    # Remove stale partial LMDBs from previous failed runs.
    if train_lmdb_dir.exists():
        shutil.rmtree(train_lmdb_dir)
    if val_lmdb_dir.exists():
        shutil.rmtree(val_lmdb_dir)

    write_parseq_lmdb(train_samples, train_lmdb_dir)
    write_parseq_lmdb(val_samples, val_lmdb_dir)


def _parse_val_accuracy_from_checkpoint(ckpt_path: str) -> Optional[float]:
    """Extract val_accuracy from checkpoint filename if available."""
    name = Path(ckpt_path).name
    if "val_accuracy=" not in name:
        return None
    try:
        segment = name.split("val_accuracy=")[1].split("-")[0]
        return float(segment)
    except Exception:
        return None


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
        f"model.batch_size=32",  # Small batch for smaller data
        f"hydra.run.dir={str(run_dir)}",
        "charset=10_digits",
        f"model.max_label_length={max_label_length}",
    ]
    
    # Add pretrained model if specified
    if pretrained_model is not None:
        pretrained_full_path = (
            Path.cwd() / pretrained_model
            if not pretrained_model.is_absolute()
            else pretrained_model
        )
        # Quote the path to handle '=' signs in checkpoint filenames
        cmd.append(f"pretrained='{str(pretrained_full_path)}'")

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
                       help="Fraction of samples for validation split (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for train/val split")
    parser.add_argument("--gt-path", type=Path, default=Path("./train/train_gt.json"),
                       help="Path to ground truth JSON file")
    parser.add_argument("--pretrained-model", type=Path, default=None,
                       help="Path to pretrained PARSeq checkpoint to fine-tune from")
    parser.add_argument("--k-folds", type=int, default=5,
                       help="Number of folds for K-fold cross-validation (default: 5)")
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
    logger.info(f"K-folds: {args.k_folds}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Filter by legibility: {args.filter_by_legibility}")
    logger.info(f"Legibility precomputed file: {args.legibility_precomputed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.legibility_precomputed is not None:
        # ------------------------------------------------------------------ #
        # Fast path: use pre-scored kept samples, skip GT loading + scoring  #
        # ------------------------------------------------------------------ #
        logger.info("\n[Step 1] Loading pre-scored kept samples (skipping GT / legibility scoring)...")
        precomputed_path = (
            Path.cwd() / args.legibility_precomputed
            if not args.legibility_precomputed.is_absolute()
            else args.legibility_precomputed
        )
        samples = load_precomputed_kept_samples(precomputed_path)
        legibility_data = {}

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

        # Step 3: Collect labeled samples
        logger.info("\n[Step 3] Collecting samples...")
        samples, legibility_data, data_stats = collect_samples(
            tracklets,
            legibility_classifier,
            filter_by_legibility=args.filter_by_legibility,
        )
        logger.info(
            "Legibility summary: total=%d kept=%d flagged_illegible=%d filtered=%d",
            data_stats['total_images_seen'],
            data_stats['images_kept_for_training'],
            data_stats['images_flagged_illegible'],
            data_stats['images_filtered_by_legibility'],
        )

    if legibility_data:
        leg_path = args.output_dir / 'legibility_scores.json'
        with open(leg_path, 'w', encoding='utf-8') as f:
            json.dump(legibility_data, f, indent=2)
        logger.info(f"Saved legibility scores to {leg_path}")

    # Step 4: Build K-fold splits
    logger.info("\n[Step 4] Building K-fold splits...")
    kfold_splits = create_kfold_splits(samples, k_folds=args.k_folds, seed=args.seed)

    # Step 5: Train PARSeq per fold
    logger.info("\n[Step 5] Training PARSeq model for each fold...")
    fold_results = []
    for fold_idx, (train_samples, val_samples) in enumerate(kfold_splits, start=1):
        fold_output_dir = args.output_dir / f"fold_{fold_idx}"
        fold_dataset_root = fold_output_dir / "parseq_data"

        logger.info("\n[FOLD %d/%d] Writing LMDB dataset...", fold_idx, args.k_folds)
        write_parseq_dataset_split(train_samples, val_samples, fold_dataset_root)

        logger.info("[FOLD %d/%d] Training...", fold_idx, args.k_folds)
        checkpoint = train_parseq(
            fold_dataset_root,
            args.epochs,
            fold_output_dir,
            args.pretrained_model,
            args.max_label_length,
        )
        val_acc = _parse_val_accuracy_from_checkpoint(checkpoint)
        fold_results.append({
            'fold': fold_idx,
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'checkpoint': checkpoint,
            'val_accuracy': val_acc,
        })
        if val_acc is not None:
            logger.info("[FOLD %d/%d] Best val_accuracy=%.4f", fold_idx, args.k_folds, val_acc)
        else:
            logger.info("[FOLD %d/%d] Completed (val_accuracy not found in checkpoint name)", fold_idx, args.k_folds)
    
    val_accs = [r['val_accuracy'] for r in fold_results if r['val_accuracy'] is not None]
    summary = {
        'k_folds': args.k_folds,
        'epochs': args.epochs,
        'num_samples_total': len(samples),
        'fold_results': fold_results,
        'mean_val_accuracy': float(np.mean(val_accs)) if val_accs else None,
        'std_val_accuracy': float(np.std(val_accs)) if val_accs else None,
    }

    if val_accs:
        best_fold = max(fold_results, key=lambda x: x['val_accuracy'] if x['val_accuracy'] is not None else float('-inf'))
        summary['best_fold'] = best_fold['fold']
        summary['best_checkpoint'] = best_fold['checkpoint']

    summary_path = args.output_dir / 'kfold_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n✅ K-fold training completed!")
    logger.info("Saved summary to %s", summary_path)
    if summary.get('mean_val_accuracy') is not None:
        logger.info(
            "K-fold val_accuracy: mean=%.4f std=%.4f",
            summary['mean_val_accuracy'],
            summary['std_val_accuracy'],
        )
    if summary.get('best_checkpoint'):
        logger.info("Best checkpoint: %s", summary['best_checkpoint'])
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
