import argparse
import logging
import pickle
import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat
from lightglue import LightGlue
from baseline.rr_baseline import load_similar_pairs, compute_precision_recall, compute_rr_baseline
from baseline.pr_baseline import compute_pr_baseline
from baseline.pr_baseline_change import compute_pr_baseline_change
from baseline.rn_baseline import compute_rn_baseline
from evaluation.pose_estimation_aachen import parse_aachen_cameras
from baseline.nn_baseline import compute_nn_baseline
from baseline.trained_matcher import load_trained_lightglu3d, load_trained_adapt, compute_trained_lightglu3d, compute_trained_lightglu3d_dynamic
from .sigma_distribution import show_results_with_sigma, show_aachen_results_fixed_sigma

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Precision/Recall for Aachen v1.1 Dataset")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Aachen root")
    parser.add_argument('--covisibility_dir', type=Path, required=True, help="Path to covisibility results")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to sfm outputs")
    parser.add_argument('--gt_path', type=Path, required=True, help="Path to aachen_ref_ground_truth.pkl")
    parser.add_argument('--method', type=str, required=True, 
                        choices=['NN', 'RR', 'RN', 'PR', 'PRC', 'TRAIN', 'ADAPT'], 
                        help="Matching method to evaluate")
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help="Path to trained network weights (Required for TRAIN/ADAPT)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = args.method
    
    logger.info(f"Starting Aachen Evaluation with Method: {method}")

    # Initialize matchers dynamically
    matchers = {}
    if method in ["RR", "RN", "PR", "PRC"]:
        logger.info("Initializing Standard LightGlue Baseline...")
        matchers['baseline'] = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    elif method == "TRAIN":
        if not args.checkpoint: raise ValueError("--checkpoint must be provided for TRAIN.")
        matchers['lightglu3d'] = load_trained_lightglu3d(args.checkpoint, device)
    elif method == "ADAPT":
        if not args.checkpoint: raise ValueError("--checkpoint must be provided for ADAPT.")
        matchers['adapt'] = load_trained_adapt(args.checkpoint, device)
    # Load pre-computed ref gt
    if not args.gt_path.exists():
        raise FileNotFoundError(f"Ground Truth file not found at {args.gt_path}.")

    logger.info("Loading HLOC Reference Ground Truth...")
    with open(args.gt_path, "rb") as f:
        gt_data = pickle.load(f)

    # Load Covisibility Graph & Most Similar Pairs
    with open(args.covisibility_dir / "covisibility_results.pkl", "rb") as f:
        covis_dict = pickle.load(f)
    pair_dict = load_similar_pairs(args.covisibility_dir / "most_similar_pairs.txt")

    # Load 3D SfM Model for Reference Poses & XYZ coordinates
    sfm_model_path = args.sfm_dir / "sfm_superpoint+lightglue"
    sfm_cameras, sfm_images, sfm_points3D = rw.read_model(sfm_model_path, ext=".bin")

    # Load query cameras
    query_cams = parse_aachen_cameras(args.dataset / "queries")

    # Data arrays for final reporting
    day_precisions, day_recalls = [], []
    night_precisions, night_recalls = [], []
    # Trackers for separate matchability (sigma)
    day_sigma0, day_sigma1 = [], []
    night_sigma0, night_sigma1 = [], []

    q_feats_path = args.sfm_dir / "feats-superpoint-n2048.h5"
    p3d_feats_path = args.covisibility_dir / "points3D_feats_cache.h5"
    
    logger.info("Opening HDF5 feature caches...")
    with h5py.File(q_feats_path, "r") as q_feats_h5, h5py.File(p3d_feats_path, "r") as p3d_feats_h5:
        
        for full_query_name, gt_matches in tqdm(gt_data.items(), desc=f"Evaluating {method} Queries"):
            if full_query_name not in covis_dict or len(covis_dict[full_query_name]["unique_points"]) == 0:
                continue
                
            visible_p3d = covis_dict[full_query_name]["unique_points"]
            
            if full_query_name not in q_feats_h5:
                continue
            
            # Extract Pre-computed GT array early to check for mathematically empty images
            gt_matches0 = gt_matches["matches0"]
            if np.sum(gt_matches0 >= 0) == 0: # Skip if GT has zero matches
                continue
            
            q_kpts = q_feats_h5[full_query_name]["keypoints"][:]
            q_desc = q_feats_h5[full_query_name]["descriptors"][:]
            q_img_size = np.array(q_feats_h5[full_query_name]["image_size"][:])

            # Get 3D features and indices
            p3d_desc, p3d_kpts = [], []
            for pid in visible_p3d:
                pid_str = str(pid)
                if pid_str in p3d_feats_h5 and int(pid) in sfm_points3D:
                    p3d_desc.append(p3d_feats_h5[pid_str]["descriptors"][:].reshape(256))
                    p3d_kpts.append(sfm_points3D[int(pid)].xyz)
                    
            if len(p3d_kpts) == 0:
                continue
                
            p3d_desc = np.vstack(p3d_desc).T 
            p3d_kpts = np.vstack(p3d_kpts)   

            # Prepare reference data for baselines
            ref_pose_matrix, ref_cam_obj = None, None
            if method in ["PR", "RR", "RN", "PRC"]:
                ref_name = pair_dict.get(full_query_name)
                if not ref_name: continue
                ref_image_obj = next((img for img in sfm_images.values() if img.name == ref_name), None)
                if not ref_image_obj: continue
                    
                ref_R = qvec2rotmat(ref_image_obj.qvec)
                ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))
                ref_cam_obj = sfm_cameras[ref_image_obj.camera_id]

            # Format q_camera for the PRC
            q_camera = None
            if method == "PRC":
                cam_obj = query_cams.get(full_query_name)
                if not cam_obj: continue
                q_camera = {
                    "intrinsics": {
                        "model": getattr(cam_obj, 'model_name', getattr(cam_obj.model, 'name', str(cam_obj.model))),
                        "width": cam_obj.width,
                        "height": cam_obj.height,
                        "params": cam_obj.params
                    }
                }

            # Get the predicted matches
            if method == "NN":
                pred_matches0 = compute_nn_baseline(q_desc, p3d_desc, device)
            elif method == "RR":
                pred_matches0, _, _, _, _ = compute_rr_baseline(matchers['baseline'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device)
            elif method == "RN":
                pred_matches0, _, _, _, _ = compute_rn_baseline(matchers['baseline'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device)
            elif method == "PR":
                pred_matches0, _, _, _, _ = compute_pr_baseline(matchers['baseline'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, ref_cam_obj, device)
            elif method == "PRC":
                pred_matches0, _, _, _, _ = compute_pr_baseline_change(matchers['baseline'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, q_camera, device)
            elif method == "TRAIN":
                pred_matches0, score_matrix = compute_trained_lightglu3d_dynamic(matchers['lightglu3d'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)
            elif method == "ADAPT":
                pred_matches0, _ = compute_trained_lightglu3d(matchers['adapt'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)
            
            # Save sigma
            if score_matrix is not None:
                dustbin_scores0 = score_matrix[:-1, -1]
                dustbin_scores1 = score_matrix[-1, :-1]
                sigma0 = 1.0 - np.exp(dustbin_scores0)
                sigma1 = 1.0 - np.exp(dustbin_scores1)
                
                if "day" in full_query_name.lower():
                    day_sigma0.append(sigma0)
                    day_sigma1.append(sigma1)
                else:
                    night_sigma0.append(sigma0)
                    night_sigma1.append(sigma1)
            
            # Compute Precision and Recall
            precision, recall, _, _, _ = compute_precision_recall(pred_matches0, gt_matches0)
            
            # If the model failed completely and predicted 0 matches, penalize it with 0.0 scores
            if precision is None: precision = 0.0
            if recall is None: recall = 0.0

            if "day" in full_query_name.lower():
                day_precisions.append(precision)
                day_recalls.append(recall)
            else:
                night_precisions.append(precision)
                night_recalls.append(recall)

    # Evaluate metrics
    logger.info("="*50)
    logger.info(f"OVERALL {method} METRICS: AACHEN v1.1")
    logger.info("="*50)
    
    if day_precisions:
        logger.info(f"[DAY]   Queries Evaluated: {len(day_precisions)}")
        logger.info(f"[DAY]   Match Precision:   {np.mean(day_precisions):.4f}")
        logger.info(f"[DAY]   Match Recall:      {np.mean(day_recalls):.4f}")
    
    if night_precisions:
        logger.info("-" * 50)
        logger.info(f"[NIGHT] Queries Evaluated: {len(night_precisions)}")
        logger.info(f"[NIGHT] Match Precision:   {np.mean(night_precisions):.4f}")
        logger.info(f"[NIGHT] Match Recall:      {np.mean(night_recalls):.4f}")
        
    all_prec = day_precisions + night_precisions
    all_rec = day_recalls + night_recalls
    
    if all_prec:
        logger.info("="*50)
        logger.info(f"[TOTAL] Queries Evaluated: {len(all_prec)}")
        logger.info(f"[TOTAL] Match Precision:   {np.mean(all_prec):.4f}")
        logger.info(f"[TOTAL] Match Recall:      {np.mean(all_rec):.4f}")
        logger.info("="*50)

    # Plot sigma distributions
    if method in ["TRAIN", "ADAPT"]:
        show_aachen_results_fixed_sigma(day_sigma0, day_sigma1, night_sigma0, night_sigma1, num_day=len(day_precisions), num_night=len(night_precisions))

if __name__ == "__main__":
    main()