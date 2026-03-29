"""
SSISv2 inference wrapper.
Detects shadow-object pairs from a single image and returns their masks.
"""

import sys
import os
import numpy as np
import cv2
import torch
from pathlib import Path

# SSISv2 repo path (relative to this file)
SSIS_ROOT = Path(__file__).parent / "SSIS"


def _setup_ssis_paths():
    """Add SSISv2 and detectron2 to sys.path."""
    ssis_root = str(SSIS_ROOT)
    if ssis_root not in sys.path:
        sys.path.insert(0, ssis_root)


def _patch_soba_registration():
    """
    SSISv2's predictor.py registers the SOBA dataset at import time.
    We patch the dataset catalog to avoid requiring SOBA annotations for inference.
    """
    try:
        from detectron2.data import DatasetCatalog, MetadataCatalog

        for split in ["SOBA_train", "SOBA_val", "SOBA_challenge"]:
            if split not in DatasetCatalog:
                DatasetCatalog.register(split, lambda: [])
                MetadataCatalog.get(split).set(
                    thing_classes=["shadow", "object"],
                    thing_colors=[(0, 0, 255), (0, 255, 0)],
                )
    except Exception:
        pass


_setup_ssis_paths()
_patch_soba_registration()


class SSISv2Detector:
    """Wrapper for SSISv2 inference."""

    def __init__(
        self,
        config_file: str = None,
        weights_path: str = None,
        confidence_threshold: float = 0.3,
        device: str = "cuda",
    ):
        from detectron2.config import get_cfg
        from adet.config import get_cfg as get_adet_cfg

        if config_file is None:
            config_file = str(
                SSIS_ROOT / "configs" / "SSIS" / "MS_R_101_BiFPN_SSISv2_demo.yaml"
            )

        if weights_path is None:
            weights_path = str(
                SSIS_ROOT
                / "tools"
                / "output"
                / "SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl"
                / "model_ssisv2_final.pth"
            )

        cfg = get_cfg()
        get_adet_cfg(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            confidence_threshold
        )
        cfg.MODEL.DEVICE = device
        cfg.freeze()

        from detectron2.engine import DefaultPredictor

        self.predictor = DefaultPredictor(cfg)
        self.confidence_threshold = confidence_threshold

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """
        Detect shadow-object pairs in a single BGR image.

        Returns:
            List of dicts, each representing a shadow-object pair:
            {
                "shadow_mask": np.ndarray (H, W) bool,
                "object_mask": np.ndarray (H, W) bool,
                "shadow_score": float,
                "object_score": float,
                "association_id": int,
            }
        """
        outputs = self.predictor(image_bgr)
        instances = outputs["instances"].to("cpu")

        if not hasattr(instances, "pred_associations"):
            return []

        masks = instances.pred_masks.numpy()  # (N, H, W)
        classes = instances.pred_classes.numpy()  # (N,) 0=shadow, 1=object
        scores = instances.scores.numpy()  # (N,)
        associations = instances.pred_associations.numpy()  # (N,)

        # Group by association ID
        pairs = {}
        for i in range(len(masks)):
            assoc_id = int(associations[i])
            if assoc_id < 0:
                continue
            if assoc_id not in pairs:
                pairs[assoc_id] = {}

            cls = int(classes[i])
            if cls == 0:  # shadow
                pairs[assoc_id]["shadow_mask"] = masks[i].astype(bool)
                pairs[assoc_id]["shadow_score"] = float(scores[i])
            elif cls == 1:  # object
                pairs[assoc_id]["object_mask"] = masks[i].astype(bool)
                pairs[assoc_id]["object_score"] = float(scores[i])

        # Filter to complete pairs only
        results = []
        for assoc_id, pair in pairs.items():
            if "shadow_mask" in pair and "object_mask" in pair:
                pair["association_id"] = assoc_id
                results.append(pair)

        # Sort by object score descending
        results.sort(key=lambda x: x["object_score"], reverse=True)
        return results


def detect_shadow_object_pairs(
    image_path: str,
    confidence_threshold: float = 0.3,
    device: str = "cuda",
) -> list[dict]:
    """
    Convenience function: detect shadow-object pairs from an image file.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    detector = SSISv2Detector(
        confidence_threshold=confidence_threshold, device=device
    )
    return detector.detect(image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SSISv2 shadow-object pair detection")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="./output_ssis", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    image = cv2.imread(args.image)
    detector = SSISv2Detector(
        confidence_threshold=args.threshold, device=args.device
    )
    pairs = detector.detect(image)

    print(f"Detected {len(pairs)} shadow-object pair(s)")
    for i, pair in enumerate(pairs):
        shadow_path = os.path.join(args.output, f"pair{i}_shadow.png")
        object_path = os.path.join(args.output, f"pair{i}_object.png")
        cv2.imwrite(shadow_path, pair["shadow_mask"].astype(np.uint8) * 255)
        cv2.imwrite(object_path, pair["object_mask"].astype(np.uint8) * 255)
        print(
            f"  Pair {i}: shadow_score={pair['shadow_score']:.3f}, "
            f"object_score={pair['object_score']:.3f}"
        )
