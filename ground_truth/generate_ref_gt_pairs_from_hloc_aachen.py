import argparse
import logging
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import h5py
from scipy.spatial import cKDTree
import pycolmap
from utils.utils import qvec2rotmat
from evaluation.pose_estimation_aachen import load_reference_poses, parse_aachen_cameras

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IGNORE_FEATURE = -2
UNMATCHED_FEATURE = -1

def compute_ground_truth_matches_aachen(
        kpts2d, pts3d, camera, gt_pose, 
        pos_reproj_thresh=3.0, neg_reproj_thresh=8.0
    ): # Skip the depth threshld
    
    N2D = kpts2d.shape[0]
    N3D = pts3d.shape[0]

    matches0 = np.full(N2D, UNMATCHED_FEATURE, dtype=int)
    matches1 = np.full(N3D, UNMATCHED_FEATURE, dtype=int)

    if N3D == 0:
        return matches0, matches1

    # HLOC Ground truth pose
    R = qvec2rotmat(gt_pose['qvec'])
    t = gt_pose['tvec'].reshape(3, 1)
    
    # Project 3D points
    X = pts3d.T 
    X_cam = R @ X + t 
    z = X_cam[2]
    valid_depth = z > 0  # Point must be in front of camera

    X_cam = X_cam[:, valid_depth]
    z = z[valid_depth]

    # Create PyCOLMAP camera for exact projection math
    colmap_cam = pycolmap.Camera(
        model=camera.model,
        width=camera.width,
        height=camera.height,
        params=camera.params
    )
    
    # Project to 2D
    projected_2d = np.array(colmap_cam.img_from_cam(X_cam.T))
    u = projected_2d[:, 0]
    v = projected_2d[:, 1]

    valid_proj = (
        (u >= 0) & (u < camera.width) &
        (v >= 0) & (v < camera.height)
    )

    u = u[valid_proj]
    v = v[valid_proj]
    valid_indices = np.where(valid_depth)[0][valid_proj]

    projected = np.stack([u, v], axis=1)
    tree = cKDTree(kpts2d)
        
    dists, min_indices = tree.query(projected, distance_upper_bound=neg_reproj_thresh)
        
    for idx3d, min_idx_2d, dist in zip(valid_indices, min_indices, dists):
        if min_idx_2d < N2D: 
            # STRICT match
            if dist <= pos_reproj_thresh:
                if matches0[min_idx_2d] in [UNMATCHED_FEATURE, IGNORE_FEATURE]:
                    matches0[min_idx_2d] = idx3d
                    matches1[idx3d] = min_idx_2d
            
            # IGNORE match (Smudged edges / slight misalignment)
            elif dist <= neg_reproj_thresh:
                if matches0[min_idx_2d] == UNMATCHED_FEATURE:
                    matches0[min_idx_2d] = IGNORE_FEATURE
                if matches1[idx3d] == UNMATCHED_FEATURE:
                    matches1[idx3d] = IGNORE_FEATURE

    return matches0, matches1

def process_aachen_gt(args):
    logger.info("Generating Aachen Pseudo-Ground Truth based on HLOC Poses...")

    gt_poses = load_reference_poses(args.hloc_reference)
    query_cams = parse_aachen_cameras(args.query_dir)
    
    with open(args.covisibility_dir / "covisibility_results.pkl", "rb") as f:
        covis_dict = pickle.load(f)

    # Open HDF5 streams
    feats_2d_h5 = h5py.File(args.sfm_dir / "feats-superpoint-n2048.h5", "r")
    feats_3d_h5 = h5py.File(args.covisibility_dir / "points3D_feats_cache.h5", "r")
    
    sfm_model_path = args.sfm_dir / "sfm_superpoint+lightglue"
    reconstruction = pycolmap.Reconstruction(sfm_model_path)

    scene_gt_data = {}
    match_counts = []
    ignore_counts = []
    
    # NEW: Trackers for Zero GT Matches
    zero_gt_day = 0
    zero_gt_night = 0

    for full_query_name in tqdm(covis_dict.keys(), desc="Processing Aachen Queries"):
        base_name = Path(full_query_name).name

        if base_name not in gt_poses or full_query_name not in query_cams:
            continue
            
        gt_pose = gt_poses[base_name]
        camera = query_cams[full_query_name]
        
        # Load 2D Keypoints
        if full_query_name not in feats_2d_h5:
            continue
        kpts2d = feats_2d_h5[full_query_name]["keypoints"][:]
        
        # Load 3D Keypoints from Covisibility graph
        visible_p3d = covis_dict[full_query_name]["unique_points"]
        pts3d = []
        for pid in visible_p3d:
            if int(pid) in reconstruction.points3D:
                pts3d.append(reconstruction.points3D[int(pid)].xyz)
        
        if not pts3d:
            continue
        pts3d = np.vstack(pts3d)

        # Generate GT
        matches0, matches1 = compute_ground_truth_matches_aachen(
            kpts2d, pts3d, camera, gt_pose, 
            args.pos_reproj_thresh, args.neg_reproj_thresh
        )
        
        scene_gt_data[full_query_name] = {
            "matches0": matches0,
            "matches1": matches1
        }
        
        num_strict_matches = np.sum(matches0 >= 0)
        match_counts.append(num_strict_matches)
        ignore_counts.append(np.sum(matches0 == IGNORE_FEATURE))
        
        # Log if the query has 0 strict matches
        if num_strict_matches == 0:
            if "day" in full_query_name.lower():
                zero_gt_day += 1
            else:
                zero_gt_night += 1

    feats_2d_h5.close()
    feats_3d_h5.close()

    output_file = args.outputs / "aachen_ref_ground_truth_6_12.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(scene_gt_data, f)

    if match_counts:
        logger.info(f"Ground Truth Generated for {len(scene_gt_data)} queries.")
        logger.info(f"Avg STRICT matches per query: {np.mean(match_counts):.2f}")
        logger.info(f"Avg IGNORED points per query: {np.mean(ignore_counts):.2f}")
        logger.info("-" * 40)
        logger.info(f"Zero GT Matches (DAY):   {zero_gt_day}")
        logger.info(f"Zero GT Matches (NIGHT): {zero_gt_night}")
        logger.info("-" * 40)
        
    logger.info(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate 2D-3D Ref-GT for Aachen from HLOC Poses")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Aachen root")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query lists")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to sfm outputs")
    parser.add_argument('--covisibility_dir', type=Path, required=True, help="Path to covisibility results")
    parser.add_argument('--outputs', type=Path, required=True, help="Directory to save GT file")
    parser.add_argument('--hloc_reference', type=Path, required=True, help="Path to Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt")
    parser.add_argument('--pos_reproj_thresh', type=float, default=6.0, help="Pixel distance for STRICT match")
    parser.add_argument('--neg_reproj_thresh', type=float, default=12.0, help="Pixel distance to IGNORE")
    # Set pos_reproj_thresh as 6.0 and neg_reproj_thresh as 12.0 based on statistics results
    args = parser.parse_args()

    args.outputs.mkdir(parents=True, exist_ok=True)
    
    try:
        process_aachen_gt(args)
    except Exception as e:
        logger.error(f"Failed to generate Aachen GT: {e}", exc_info=True)

if __name__ == "__main__":
    main()