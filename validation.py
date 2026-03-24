#!/usr/bin/env python3
"""
Validation script: Test trained PARSeq model on images NOT in the legible training set.
"""

import json
import argparse
import sys
from pathlib import Path
import random
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parseq to path
sys.path.insert(0, str(Path("parseq-main").resolve()))

from strhub.models.parseq.system import PARSeq
from strhub.models.utils import load_from_checkpoint
from strhub.data.module import SceneTextDataModule


def build_ground_truth_map(gt_path: Path, image_dir: Path):
    """
    Build a mapping of image paths to labels.
    """
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    
    result = {}
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')
    for folder, label in gt.items():
        folder_path = image_dir / str(folder)
        if folder_path.exists():
            for pattern in exts:
                for img_file in folder_path.glob(pattern):
                    result[str(img_file.resolve())] = str(label)
    
    return result


def load_legible_samples(kept_samples_path: Path) -> set:
    """Load the set of paths that were used in training."""
    legible_paths = set()
    with open(kept_samples_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                path = Path(parts[0]).resolve()
                legible_paths.add(str(path))
    return legible_paths


def load_checkpoint(ckpt_path: Path):
    """Load trained model from checkpoint using the official strhub loader."""
    model = load_from_checkpoint(str(ckpt_path)).eval()
    logger.info("Checkpoint loaded via strhub loader (charset_train=%s, max_label_length=%d)",
                model.hparams.charset_train, model.hparams.max_label_length)
    return model


def predict_on_image(model: nn.Module, image_path: str, img_transform, max_pred_len: int = 2) -> str:
    """
    Run inference on a single image.
    
    Returns:
        Predicted text string (empty if error)
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        
        # Use PARSeq's own transform pipeline for consistency with training/eval.
        img_tensor = img_transform(img).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
                model = model.cuda()
            
            # Get logits from model
            logits = model(img_tensor)
            
            # Decode output using PARSeq tokenizer + EOS handling
            pred = decode_logits(logits, model, max_pred_len=max_pred_len)
        
        return pred
    except Exception as e:
        logger.debug(f"Error predicting on {image_path}: {e}")
        return ""


def decode_logits(logits, model, max_pred_len: int = 2):
    """Decode PARSeq logits with tokenizer/EOS and clamp to numeric jersey length."""
    # Slice to [batch, max_label_length+1, charset+EOS] — mirrors str.py's [:, :3, :11].
    # This drops BOS (idx -2) and PAD (idx -1) tokens which should never be predicted.
    n_pos = model.hparams.max_label_length + 1  # e.g. 3 for max_label_length=2
    n_tok = len(model.hparams.charset_train) + 1  # charset + EOS
    logits = logits[:, :n_pos, :n_tok]
    probs = F.softmax(logits, dim=-1)
    preds, _ = model.tokenizer.decode(probs)
    text = preds[0] if preds else ""

    # Keep only digits and clamp long strings (jersey numbers are short).
    text = ''.join(ch for ch in text if ch.isdigit())
    if max_pred_len > 0:
        text = text[:max_pred_len]

    return text.strip()


def character_error_rate(pred: str, gt: str) -> float:
    """Calculate character error rate."""
    if not gt:
        return 1.0 if pred else 0.0
    
    pred = pred.upper().strip()
    gt = gt.upper().strip()
    
    if pred == gt:
        return 0.0
    
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, pred, gt)
    ratio = matcher.ratio()
    return 1.0 - ratio


def normalized_edit_distance(pred: str, gt: str) -> float:
    """Normalized edit distance (Levenshtein)."""
    if not gt:
        return 0.0 if not pred else 1.0
    
    pred = pred.upper().strip()
    gt = gt.upper().strip()
    
    if pred == gt:
        return 0.0
    
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == gt[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / max(m, n)


def tracklet_vote(results: list) -> dict:
    """
    Aggregate per-frame predictions into per-tracklet majority-vote predictions.
    Tracklet ID is the parent directory name of each image path.
    Returns dict mapping tracklet_id -> {'gt': str, 'pred': str, 'correct': bool}.
    """
    from collections import defaultdict, Counter
    tracklets = defaultdict(list)  # tracklet_id -> list of preds
    tracklet_gt = {}               # tracklet_id -> gt label
    for r in results:
        tkl = Path(r['image']).parent.name
        tracklets[tkl].append(r['pred'])
        tracklet_gt[tkl] = r['gt']

    out = {}
    for tkl, preds in tracklets.items():
        # Majority vote; prefer non-empty; ties broken by first occurrence
        non_empty = [p for p in preds if p]
        winner = Counter(non_empty).most_common(1)[0][0] if non_empty else ''
        gt = tracklet_gt[tkl]
        out[tkl] = {'gt': gt, 'pred': winner, 'correct': winner.upper() == gt.upper()}
    return out


def validate(
    checkpoint_path: Path,
    gt_path: Path,
    kept_samples_path: Optional[Path],
    image_dir: Path,
    num_samples: int = 100,
    seed: int = 42,
    max_pred_len: int = 2,
    include_unknown: bool = False,
    tracklet_vote_enabled: bool = False,
):
    """
    Validate model on images NOT in the legible training set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info("=" * 80)
    logger.info("PARSeq Validation on Non-Legible Images")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\n[Step 1] Loading ground truth and (optional) legible samples...")
    image_dir = Path(image_dir).resolve()
    gt_map = build_ground_truth_map(gt_path, image_dir)
    legible_paths = set()
    if kept_samples_path is not None:
        kept_samples_path = Path(kept_samples_path)
        if kept_samples_path.exists():
            legible_paths = load_legible_samples(kept_samples_path)
        else:
            logger.warning("  kept-samples file not found: %s (continuing without filtering)", kept_samples_path)
    
    total_with_gt = len(gt_map)
    if not include_unknown:
        gt_map = {img_path: label for img_path, label in gt_map.items() if str(label).strip() != '-1'}
    skipped_unknown = total_with_gt - len(gt_map)

    logger.info(f"  Total images with GT labels: {total_with_gt}")
    logger.info(f"  Unknown labels skipped (-1): {skipped_unknown}")
    logger.info(f"  Legible images loaded for filtering: {len(legible_paths)}")
    
    # Build candidate validation set
    if legible_paths:
        candidate_images = []
        for img_path, label in gt_map.items():
            if img_path not in legible_paths:
                candidate_images.append((img_path, label))
        logger.info(f"  Non-legible images available for validation: {len(candidate_images)}")
    else:
        candidate_images = list(gt_map.items())
        logger.info(f"  Using all GT images for validation candidates: {len(candidate_images)}")

    # Sample validation set
    if len(candidate_images) < num_samples:
        logger.warning(f"Only {len(candidate_images)} candidate images available, using all")
        val_samples = candidate_images
    else:
        val_samples = random.sample(candidate_images, num_samples)
    
    logger.info(f"\n[Step 2] Loading model from checkpoint: {checkpoint_path.name}")
    model = load_checkpoint(checkpoint_path)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    logger.info(f"  ✓ Model loaded successfully")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info(f"  ✓ Model moved to CUDA")
    
    logger.info(f"\n[Step 3] Running inference on {len(val_samples)} validation images...")
    
    results = []
    correct = 0
    cer_scores = []
    ned_scores = []
    
    for img_path, gt_label in tqdm(val_samples, desc="Validating"):
        # Predict
        pred = predict_on_image(model, img_path, img_transform, max_pred_len=max_pred_len)
        
        # Metrics
        is_correct = (pred.upper() == gt_label.upper())
        correct += is_correct
        
        cer = character_error_rate(pred, gt_label)
        ned = normalized_edit_distance(pred, gt_label)
        
        cer_scores.append(cer)
        ned_scores.append(ned)
        
        results.append({
            'image': img_path,
            'gt': gt_label,
            'pred': pred,
            'correct': is_correct,
            'cer': cer,
            'ned': ned,
        })
    
    # Compute metrics
    accuracy = 100.0 * correct / len(val_samples) if len(val_samples) > 0 else 0.0
    avg_cer = np.mean(cer_scores) if cer_scores else 0.0
    avg_ned = np.mean(ned_scores) if ned_scores else 0.0
    
    logger.info(f"\n[Step 4] Results Summary")
    logger.info("=" * 80)
    logger.info(f"  Validation samples tested: {len(val_samples)}")
    logger.info(f"  Correct predictions: {correct}/{len(val_samples)}")
    logger.info(f"  Accuracy (frame-level): {accuracy:.2f}%")
    logger.info(f"  Avg Character Error Rate (CER): {avg_cer:.4f}")
    logger.info(f"  Avg Normalized Edit Distance (NED): {avg_ned:.4f}")

    # Tracklet-level voting
    vote_results = tracklet_vote(results)
    tkl_correct = sum(1 for v in vote_results.values() if v['correct'])
    tkl_total = len(vote_results)
    tkl_acc = 100.0 * tkl_correct / tkl_total if tkl_total else 0.0
    logger.info(f"  Tracklets evaluated: {tkl_total}")
    logger.info(f"  Correct tracklets (majority vote): {tkl_correct}/{tkl_total}")
    logger.info(f"  Accuracy (tracklet-level): {tkl_acc:.2f}%")
    logger.info("=" * 80)
    
    # Show sample results
    logger.info(f"\n[Sample Predictions (first 10)]:")
    logger.info(f"  {'Status':8} {'Image':45} {'GT':15} {'Pred':15}")
    logger.info(f"  {'-'*8} {'-'*45} {'-'*15} {'-'*15}")
    for i, res in enumerate(results[:10]):
        status = "✓ PASS" if res['correct'] else "✗ FAIL"
        img_name = Path(res['image']).name
        logger.info(f"  {status:8} {img_name:45} {res['gt']:15} {res['pred']:15}")
    
    # Save detailed results
    output_path = Path("validation_results.json")
    vote_list = [{'tracklet': tkl, **v} for tkl, v in vote_results.items()]
    with open(output_path, 'w') as f:
        json.dump({
            'checkpoint': str(checkpoint_path),
            'num_samples': len(val_samples),
            'accuracy_frame_pct': accuracy,
            'accuracy_tracklet_pct': tkl_acc,
            'avg_cer': avg_cer,
            'avg_ned': avg_ned,
            'results': results,
            'tracklet_results': vote_list,
        }, f, indent=2)
    logger.info(f"\n✓ Detailed results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate trained PARSeq model on validation/test images")
    parser.add_argument("--checkpoint", type=Path, 
                       default=Path("outputs/parseq_finetuned_from_95pct/train_run/checkpoints/epoch=17-step=1404-val_accuracy=97.7456-val_NED=98.2287.ckpt"),
                       help="Path to trained checkpoint")
    parser.add_argument("--gt-path", type=Path, default=Path("test/test_gt.json"),
                       help="Path to ground truth JSON")
    parser.add_argument("--kept-samples", type=Path, 
                       default=None,
                       help="Optional path to legibility_kept_samples_legible.txt from training")
    parser.add_argument("--image-dir", type=Path, default=Path("test"),
                       help="Root directory of images")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of validation samples to test")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--max-pred-len", type=int, default=2,
                       help="Maximum predicted length to keep (default: 2 for jersey numbers)")
    parser.add_argument("--include-unknown", action="store_true",
                       help="Include unknown GT labels (-1) in metrics")
    parser.add_argument("--tracklet-vote", action="store_true",
                       help="Report tracklet-level majority-vote accuracy in addition to frame-level")
    
    args = parser.parse_args()
    
    validate(
        checkpoint_path=args.checkpoint,
        gt_path=args.gt_path,
        kept_samples_path=args.kept_samples,
        image_dir=args.image_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        max_pred_len=args.max_pred_len,
        include_unknown=args.include_unknown,
        tracklet_vote_enabled=args.tracklet_vote,
    )
