import argparse
import logging
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
from .generate_gt_pairs_by_scene import load_query_cams, generate_gt_for_query_soft

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_FEATURE = -2
UNMATCHED_FEATURE = -1

def process_scene(scene_path: Path, args):
    scene_name = scene_path.name
    logger.info(f"Generating Ground Truth Pairs for {scene_name}...")

    query_path = args.query_dir / scene_name
    query_names_file = query_path / "query_image_names.txt"
    query_pose_file = query_path / "query_image_cameras.txt"
    feats_3d_path = args.feature_dir / scene_name / "points3D_feats_cache.h5"
    feats_2d_path = args.sfm_dir / scene_name / "feats-superpoint-n2048.h5"
    covisibility_path = args.feature_dir / scene_name / "covisibility_results.pkl"
    depth_path = args.depth_dir / scene_name
    output_file = args.feature_dir / scene_name / "ground_truth.pkl"
    
    if not all([query_names_file.exists(), feats_3d_path.exists(), covisibility_path.exists()]):
        logger.error(f"Missing required input files for {scene_name}. Skipping.")
        return

    with open(query_names_file, 'r') as f:
        query_list = [line.strip() for line in f]

    with open(covisibility_path, "rb") as f:
        covisibility_dict = pickle.load(f)

    query_cams = load_query_cams(query_pose_file)
    scene_gt_data = {}

    match_counts = []
    ignore_counts = []

    for query in tqdm(query_list, desc=f"Processing Queries in {scene_name}"):
        gt_data = generate_gt_for_query_soft(
            query, feats_2d_path, feats_3d_path, query_cams, covisibility_dict, depth_path, args
        )
        if gt_data is not None:
            scene_gt_data[query] = gt_data
            # Tally strict matches (>= 0) and ignored features (-2)
            match_counts.append(np.sum(gt_data["matches0"] >= 0))
            ignore_counts.append(np.sum(gt_data["matches0"] == IGNORE_FEATURE))

    with open(output_file, "wb") as f:
        pickle.dump(scene_gt_data, f)

    if match_counts:
        logger.info(f"[{scene_name}] Ground Truth Generated for {len(scene_gt_data)} queries.")
        logger.info(f"[{scene_name}] Avg STRICT matches per query: {np.mean(match_counts):.2f}")
        logger.info(f"[{scene_name}] Avg IGNORED (-2) points per query: {np.mean(ignore_counts):.2f}")
    logger.info(f"Saved to {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate 2D-3D Ground Truth Matches (LightGlue Logic)")
    parser.add_argument('--depth_dir', type=Path, required=True, help="Path to depth_undistorted")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query_sets")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to sfm outputs")
    parser.add_argument('--feature_dir', type=Path, required=True, help="Path to feature_results (covisibility & 3D feats)")
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--scene_list', type=Path, default=None)
    parser.add_argument('--pos_reproj_thresh', type=float, default=3.0, help="Pixel distance for STRICT match")
    parser.add_argument('--neg_reproj_thresh', type=float, default=5.0, help="Pixel distance to IGNORE (beyond is Unmatchable)")
    parser.add_argument('--pos_depth_thresh', type=float, default=0.10, help="Relative depth error for STRICT match (10%)")
    parser.add_argument('--neg_depth_thresh', type=float, default=0.25, help="Relative depth error to IGNORE (25%)")
    args = parser.parse_args()

    scenes = []
    if args.scene_list:
        if not args.scene_list.exists():
            raise FileNotFoundError(f"Scene list file not found: {args.scene_list}")
        logger.info(f"Reading scenes from {args.scene_list}...")
        with open(args.scene_list, 'r') as f:
            scene_names = [line.strip() for line in f if line.strip()]
        for name in scene_names:
            scenes.append(args.depth_dir / name)
    elif args.scene:
        scenes = [args.depth_dir / args.scene]
    else:
        scenes = sorted([p for p in args.depth_dir.iterdir() if p.is_dir()])

    logger.info(f"Found {len(scenes)} scenes to process.")

    for scene_path in scenes:
        try:
            process_scene(scene_path, args)
        except Exception as e:
            logger.error(f"Failed to process {scene_path.name}: {e}", exc_info=True)
            continue

if __name__ == "__main__":
    main()