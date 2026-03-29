"""
Method 3 Pipeline: SSISv2 (keyframe) + SAM2 Video Predictor (propagation)

Flow:
  1. Extract video frames to JPEG directory
  2. Run SSISv2 on keyframe(s) to detect shadow-object pairs
  3. Feed detected masks into SAM2 Video Predictor
  4. Propagate masks across all frames
  5. Save results (masks + pair metadata)
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path


def extract_frames(video_path: str, output_dir: str) -> tuple[int, float]:
    """Extract video frames as JPEG files. Returns (total frame count, fps)."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir} (fps={fps:.2f})")
    return frame_count, fps


def write_video(frames_dir: str, output_path: str, fps: float):
    """Write JPEG frames to an H.264 MP4 video file using ffmpeg."""
    import subprocess

    frame_pattern = os.path.join(frames_dir, "%05d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def save_masks(
    output_dir: str,
    video_segments: dict[int, dict[int, np.ndarray]],
    pair_info: list[dict],
):
    """Save propagated masks and pair metadata."""
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    for frame_idx, obj_masks in sorted(video_segments.items()):
        for obj_id, mask in obj_masks.items():
            mask_path = os.path.join(masks_dir, f"frame{frame_idx:05d}_obj{obj_id}.png")
            cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)

    # Save pair metadata
    meta = {
        "pairs": [],
        "total_frames": len(video_segments),
    }
    for pair in pair_info:
        entry = {
            "shadow_obj_id": pair["shadow_obj_id"],
            "object_obj_id": pair["object_obj_id"],
            "shadow_score": pair["shadow_score"],
            "object_score": pair["object_score"],
        }
        if "keyframe" in pair:
            entry["keyframe"] = pair["keyframe"]
        meta["pairs"].append(entry)

    meta_path = os.path.join(output_dir, "pairs.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved masks to {masks_dir}")
    print(f"Saved pair metadata to {meta_path}")


def visualize_frame(
    image_bgr: np.ndarray,
    obj_masks: dict[int, np.ndarray],
    pair_info: list[dict],
) -> np.ndarray:
    """Overlay masks on image with color-coded pairs."""
    vis = image_bgr.copy()

    # Color palette: each pair gets a hue, shadow=dark, object=bright
    pair_colors = {}
    for i, pair in enumerate(pair_info):
        hue = int(180 * i / max(len(pair_info), 1))
        pair_colors[pair["shadow_obj_id"]] = np.array(
            cv2.cvtColor(
                np.uint8([[[hue, 150, 120]]]), cv2.COLOR_HSV2BGR
            )[0, 0],
            dtype=np.uint8,
        )
        pair_colors[pair["object_obj_id"]] = np.array(
            cv2.cvtColor(
                np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR
            )[0, 0],
            dtype=np.uint8,
        )

    for obj_id, mask in obj_masks.items():
        if obj_id in pair_colors:
            color = pair_colors[obj_id]
            overlay = vis.copy()
            overlay[mask] = color
            vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    return vis


def _compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _get_keyframe_indices(
    frame_count: int,
    keyframes: list[int] | None,
    keyframe_interval: int | None,
) -> list[int]:
    """Determine which frames to run SSISv2 on."""
    if keyframes is not None:
        return sorted(set(kf for kf in keyframes if 0 <= kf < frame_count))
    if keyframe_interval is not None and keyframe_interval > 0:
        return list(range(0, frame_count, keyframe_interval))
    return [0]


def run_pipeline(
    video_path: str,
    output_dir: str,
    keyframe_idx: int = 0,
    keyframes: list[int] | None = None,
    keyframe_interval: int | None = None,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.3,
    sam2_model_size: str = "large",
    device: str = "cuda",
    save_visualization: bool = True,
):
    """
    Run the full SSISv2 + SAM2 pipeline.

    Args:
        video_path: Path to input video file.
        output_dir: Directory to save results.
        keyframe_idx: Frame index for SSISv2 detection (single keyframe, legacy).
        keyframes: List of frame indices for SSISv2 detection.
        keyframe_interval: Run SSISv2 every N frames (alternative to keyframes).
        confidence_threshold: SSISv2 detection threshold.
        iou_threshold: IoU threshold to consider a detection as duplicate.
        sam2_model_size: SAM2 model size (tiny/small/base_plus/large).
        device: CUDA device.
        save_visualization: Whether to save visualized overlay frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")

    # --- Step 1: Extract frames ---
    print("=" * 50)
    print("[Step 1/4] Extracting video frames...")
    frame_count, fps = extract_frames(video_path, frames_dir)

    # Determine keyframes
    if keyframes is None and keyframe_interval is None:
        kf_indices = [keyframe_idx]
    else:
        kf_indices = _get_keyframe_indices(frame_count, keyframes, keyframe_interval)

    for kf in kf_indices:
        if kf >= frame_count:
            raise ValueError(f"keyframe={kf} exceeds frame count={frame_count}")

    print(f"  Keyframes: {kf_indices}")

    # --- Step 2: SSISv2 detection on keyframes ---
    print("=" * 50)
    print(f"[Step 2/4] Running SSISv2 on {len(kf_indices)} keyframe(s)...")
    from ssis_inference import SSISv2Detector

    detector = SSISv2Detector(
        confidence_threshold=confidence_threshold, device=device
    )

    # Detect on each keyframe and deduplicate
    next_pair_id = 0
    keyframe_masks = {}   # {frame_idx: {obj_id: mask}}
    pair_info = []        # list of pair dicts
    all_object_masks = [] # track all assigned object masks for dedup

    for kf_idx in kf_indices:
        kf_path = os.path.join(frames_dir, f"{kf_idx:05d}.jpg")
        kf_image = cv2.imread(kf_path)
        pairs = detector.detect(kf_image)
        print(f"  Frame {kf_idx}: detected {len(pairs)} pair(s)")

        new_masks = {}
        for pair in pairs:
            # Check if this pair's object overlaps with any already-tracked object
            is_duplicate = False
            for existing_mask in all_object_masks:
                iou = _compute_mask_iou(pair["object_mask"], existing_mask)
                if iou >= iou_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                print(f"    Skipping duplicate pair (IoU >= {iou_threshold:.2f})")
                continue

            shadow_id = next_pair_id * 2
            object_id = next_pair_id * 2 + 1
            new_masks[shadow_id] = pair["shadow_mask"]
            new_masks[object_id] = pair["object_mask"]
            pair_info.append(
                {
                    "shadow_obj_id": shadow_id,
                    "object_obj_id": object_id,
                    "shadow_score": pair["shadow_score"],
                    "object_score": pair["object_score"],
                    "keyframe": kf_idx,
                }
            )
            all_object_masks.append(pair["object_mask"])
            print(
                f"    Pair {next_pair_id}: shadow(id={shadow_id}) + object(id={object_id}) "
                f"scores=({pair['shadow_score']:.3f}, {pair['object_score']:.3f}) "
                f"[keyframe={kf_idx}]"
            )
            next_pair_id += 1

        if new_masks:
            keyframe_masks[kf_idx] = new_masks

    total_pairs = len(pair_info)
    print(f"  Total unique pairs: {total_pairs}")
    if total_pairs == 0:
        print("  No pairs detected. Try lowering --threshold or adjusting keyframes.")
        return

    # --- Step 3: SAM2 propagation ---
    print("=" * 50)
    print("[Step 3/4] Propagating masks with SAM2 Video Predictor...")
    from sam2_propagate import SAM2VideoPropagator

    propagator = SAM2VideoPropagator(model_size=sam2_model_size, device=device)
    video_segments = propagator.propagate_multi(frames_dir, keyframe_masks)

    print(f"  Propagated to {len(video_segments)} frames")

    # --- Save results ---
    print("=" * 50)
    print("[Step 4/4] Saving results...")
    save_masks(output_dir, video_segments, pair_info)

    # --- Optional: save visualization ---
    if save_visualization:
        vis_dir = os.path.join(output_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        for frame_idx in sorted(video_segments.keys()):
            frame_path = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
            frame_image = cv2.imread(frame_path)
            vis = visualize_frame(frame_image, video_segments[frame_idx], pair_info)
            vis_path = os.path.join(vis_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(vis_path, vis)
        print(f"  Saved visualization to {vis_dir}")

        # Write visualization video
        vis_video_path = os.path.join(output_dir, "visualization.mp4")
        write_video(vis_dir, vis_video_path, fps)
        print(f"  Saved visualization video to {vis_video_path}")

    print("=" * 50)
    print("Done!")
    print(f"  Pairs: {len(pair_info)}")
    print(f"  Frames: {len(video_segments)}")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="SSISv2 + SAM2 Video Shadow-Object Detection Pipeline"
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument(
        "--keyframe", type=int, default=0, help="Keyframe index for SSISv2 detection (single)"
    )
    parser.add_argument(
        "--keyframes", type=str, default=None,
        help="Comma-separated keyframe indices (e.g., '0,30,60'). Overrides --keyframe.",
    )
    parser.add_argument(
        "--keyframe-interval", type=int, default=None,
        help="Run SSISv2 every N frames (e.g., 30). Overrides --keyframe.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="SSISv2 confidence threshold"
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.3,
        help="IoU threshold for deduplicating detections across keyframes",
    )
    parser.add_argument(
        "--sam2-model",
        default="large",
        choices=["tiny", "small", "base_plus", "large"],
        help="SAM2 model size",
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--no-vis", action="store_true", help="Skip saving visualization"
    )
    args = parser.parse_args()

    keyframes = None
    if args.keyframes is not None:
        keyframes = [int(x.strip()) for x in args.keyframes.split(",")]

    run_pipeline(
        video_path=args.video,
        output_dir=args.output,
        keyframe_idx=args.keyframe,
        keyframes=keyframes,
        keyframe_interval=args.keyframe_interval,
        confidence_threshold=args.threshold,
        iou_threshold=args.iou_threshold,
        sam2_model_size=args.sam2_model,
        device=args.device,
        save_visualization=not args.no_vis,
    )


if __name__ == "__main__":
    main()
