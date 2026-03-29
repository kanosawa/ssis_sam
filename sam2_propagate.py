"""
SAM2 Video Predictor wrapper.
Takes initial masks and propagates them across all video frames.
"""

import os
import numpy as np
import torch
from pathlib import Path

SAM2_ROOT = Path(__file__).parent / "sam2"

# Default checkpoint paths
SAM2_CONFIGS = {
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
}

SAM2_CHECKPOINTS = {
    "large": "checkpoints/sam2.1_hiera_large.pt",
    "base_plus": "checkpoints/sam2.1_hiera_base_plus.pt",
    "small": "checkpoints/sam2.1_hiera_small.pt",
    "tiny": "checkpoints/sam2.1_hiera_tiny.pt",
}


class SAM2VideoPropagator:
    """Wrapper for SAM2 video mask propagation."""

    def __init__(
        self,
        model_size: str = "large",
        device: str = "cuda",
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ):
        self.device = torch.device(device)
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu

        config_rel = SAM2_CONFIGS[model_size]
        checkpoint_path = str(SAM2_ROOT / SAM2_CHECKPOINTS[model_size])

        # Initialize Hydra with sam2's config directory
        # Configs live inside the sam2 Python package: sam2/sam2/configs/
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        # config_rel is like "configs/sam2.1/sam2.1_hiera_l.yaml"
        # The actual configs are at SAM2_ROOT/sam2/configs/sam2.1/
        config_dir = str(SAM2_ROOT / "sam2" / os.path.dirname(config_rel))
        config_name = os.path.basename(config_rel)

        from sam2.build_sam import build_sam2_video_predictor

        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            self.predictor = build_sam2_video_predictor(
                config_name, checkpoint_path, device=self.device
            )

    def propagate(
        self,
        frames_dir: str,
        initial_masks: dict[int, np.ndarray],
        keyframe_idx: int = 0,
    ) -> dict[int, dict[int, np.ndarray]]:
        """
        Propagate initial masks across all video frames (single keyframe).

        Args:
            frames_dir: Directory containing JPEG frames (00000.jpg, 00001.jpg, ...)
            initial_masks: {obj_id: mask} where mask is (H, W) bool array.
            keyframe_idx: Frame index where initial_masks are defined.

        Returns:
            {frame_idx: {obj_id: mask}} for all frames.
        """
        return self.propagate_multi(frames_dir, {keyframe_idx: initial_masks})

    def propagate_multi(
        self,
        frames_dir: str,
        keyframe_masks: dict[int, dict[int, np.ndarray]],
    ) -> dict[int, dict[int, np.ndarray]]:
        """
        Propagate masks from multiple keyframes across all video frames.

        Args:
            frames_dir: Directory containing JPEG frames (00000.jpg, 00001.jpg, ...)
            keyframe_masks: {frame_idx: {obj_id: mask}}
                Each mask is a (H, W) bool array.

        Returns:
            {frame_idx: {obj_id: mask}} for all frames.
            Each mask is a (H, W) bool numpy array.
        """
        with torch.inference_mode(), torch.autocast(
            str(self.device), dtype=torch.bfloat16
        ):
            inference_state = self.predictor.init_state(
                video_path=frames_dir,
                offload_video_to_cpu=self.offload_video_to_cpu,
                offload_state_to_cpu=self.offload_state_to_cpu,
            )

            # Add masks at each keyframe
            for kf_idx, masks in keyframe_masks.items():
                for obj_id, mask in masks.items():
                    self.predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=kf_idx,
                        obj_id=obj_id,
                        mask=mask,
                    )

            # Propagate forward through the video
            video_segments = {}
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                inference_state
            ):
                video_segments[frame_idx] = {}
                for i, obj_id in enumerate(obj_ids):
                    video_segments[frame_idx][int(obj_id)] = (
                        (mask_logits[i] > 0.0).squeeze(0).cpu().numpy()
                    )

            # Propagate backward to cover frames before later keyframes
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                inference_state, reverse=True
            ):
                if frame_idx not in video_segments:
                    video_segments[frame_idx] = {}
                for i, obj_id in enumerate(obj_ids):
                    oid = int(obj_id)
                    # Backward results fill in gaps; don't overwrite forward results
                    if oid not in video_segments[frame_idx]:
                        video_segments[frame_idx][oid] = (
                            (mask_logits[i] > 0.0).squeeze(0).cpu().numpy()
                        )

        return video_segments


def propagate_masks(
    frames_dir: str,
    initial_masks: dict[int, np.ndarray],
    keyframe_idx: int = 0,
    model_size: str = "large",
    device: str = "cuda",
) -> dict[int, dict[int, np.ndarray]]:
    """Convenience function for mask propagation."""
    propagator = SAM2VideoPropagator(model_size=model_size, device=device)
    return propagator.propagate(frames_dir, initial_masks, keyframe_idx)
