import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm
from hloc.utils import read_write_model as rw
from collections import defaultdict
import os

import argparse
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_3d_descriptors(points3D, images, h5_path):
    """
    Compute the averaged features for 3D points
    Returns: { p3d_id: {'descriptors': ..., 'keypoints': ..., 'scores': ...} }
    """
    # First load image <-> 3D point correspondences in to dict
    image_to_obs = defaultdict(list)

    for p3d_id, p3d_obj in points3D.items():
        for img_id, p2d_idx in zip(p3d_obj.image_ids, p3d_obj.point2D_idxs):
            if img_id in images:
                image_to_obs[img_id].append((p3d_id, p2d_idx))

    p3d_feature_dict = defaultdict(lambda: {
        "descriptors": [],
        "scores": [],
        "xyz": None
    })

    # Loop over image ids and inner loop for current image cooresponding 3Ds
    with h5py.File(h5_path, "r") as f_h5:
        for img_id, observations in tqdm(image_to_obs.items()):

            img_name = images[img_id].name
            if img_name not in f_h5:
                logger.warning(f"{img_name} not found in {h5_path}")
                continue

            ds = f_h5[img_name]
            descriptors = ds["descriptors"][:]
            scores = ds["scores"][:]

            for p3d_id, p2d_idx in observations:
                p3d_feature_dict[p3d_id]["descriptors"].append(
                    descriptors[:, p2d_idx]
                )
                p3d_feature_dict[p3d_id]["scores"].append(
                    scores[p2d_idx]
                )
                p3d_feature_dict[p3d_id]["xyz"] = points3D[p3d_id].xyz

    final_dict = {}

    for p3d_id, data in p3d_feature_dict.items():
        descs = np.stack(data["descriptors"], axis=0)
        avg_desc = descs.mean(axis=0)
        avg_desc /= (np.linalg.norm(avg_desc) + 1e-6)

        avg_score = np.mean(data["scores"])

        final_dict[str(p3d_id)] = {
            "descriptors": avg_desc.reshape(1, -1), # shape(num, 256)
            "keypoints": data["xyz"].reshape(1, 3),
            "scores": np.array([avg_score])
        }
    logger.info("Finished 3D features computation.")

    return final_dict

def save_3d_features_to_h5(feature_dict: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, "w") as f:
        for p3d_id, data in feature_dict.items():
            grp = f.create_group(p3d_id)
            grp.create_dataset('descriptors', data=data['descriptors'])
            grp.create_dataset('keypoints', data=data['keypoints'])
            grp.create_dataset('scores', data=data['scores'])

def process_scene(scene_path: Path, args):
    scene_name = scene_path.name
    logger.info(f"Start averaged feature computation for scene: {scene_name}...")
    
    # Define paths
    sfm_model_path = args.sfm_dir / scene_name / "sfm_superpoint+lightglue"
    features_h5_path = args.sfm_dir / scene_name / "feats-superpoint-n2048.h5"
    output_dir = args.outputs / scene_name
    cached_feats_path = output_dir / "points3D_feats_cache.h5"
    if not sfm_model_path.exists():
        logger.warning(f"Skipping {scene_name}: SfM model not found at {sfm_model_path}")
        return
    if not features_h5_path.exists():
        logger.warning(f"Skipping {scene_name}: 2D Features not found at {features_h5_path}")
        return
        
    if cached_feats_path.exists():
        logger.info(f"Skipping {scene_name}: Features already computed at {cached_feats_path}")
        return

    _, images, points3D = rw.read_model(sfm_model_path, ext=".bin")
    p3d_feats = extract_3d_descriptors(points3D, images, features_h5_path)
    logger.info(f"Extracted features for {len(p3d_feats)} 3D points in {scene_name}.")
    save_3d_features_to_h5(p3d_feats, cached_feats_path)
    logger.info(f"Averaged feature for scene {scene_name} saved to {cached_feats_path}.")

def main():
    parser = argparse.ArgumentParser(description="Precompute 3D SuperPoint Features")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Undistorted_SfM")
    parser.add_argument('--outputs', type=Path, required=True, help="Path to output midterm_results directory")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to the directory containing triangulated models")
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--scene_list', type=Path, default=None)
    args = parser.parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset}")

    scenes = []
    if args.scene_list:
        if not args.scene_list.exists():
            raise FileNotFoundError(f"Scene list file not found: {args.scene_list}")
        logger.info(f"Reading scenes from {args.scene_list}...")
        with open(args.scene_list, 'r') as f:
            scene_names = [line.strip() for line in f if line.strip()]
        for name in scene_names:
            scenes.append(args.dataset / name)
    elif args.scene:
        scenes = [args.dataset / args.scene]
    else:
        scenes = sorted([p for p in args.dataset.iterdir() if p.is_dir() and (p / "images").exists()])

    logger.info(f"Found {len(scenes)} scenes to process.")

    for scene_path in scenes:
        try:
            process_scene(scene_path, args)
        except Exception as e:
            logger.error(f"Failed to process {scene_path.name}: {e}", exc_info=True)
            continue

if __name__ == "__main__":
    main()