# For train: TRAIN(Lightglu3d two self and one bidirectional cross), 
#            ADAPT(lightglue+adapter)

# Before use it, add the gluefactory path in the terminal
# export PYTHONPATH="/home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match:$PYTHONPATH"

import argparse
import logging
import pickle
import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm
from ground_truth.generate_gt_pairs_by_scene import load_query_cams, compute_ground_truth_matches_soft
from visualization.visualize_matches import load_trained_lightglu3d, load_trained_adapt, compute_trained_lightglu3d
from baseline.rr_baseline import load_similar_pairs, compute_precision_recall
from .sigma_distribution import show_results_with_sigma

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Precision/Recall for TRAIN and ADAPT methods across scenes")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Undistorted_SfM")
    parser.add_argument('--covisibility_dir', type=Path, required=True, help="Path to covisibility")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to sfm outputs")
    parser.add_argument('--depth_dir', type=Path, required=True, help="Path to depth maps")
    parser.add_argument('--scene_list', type=Path, required=True, help="Path to text file containing list of scenes")
    parser.add_argument('--method', type=str, required=True, choices=['TRAIN', 'ADAPT'], 
                        help="Matching method to evaluate: TRAIN or ADAPT")
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help="Path to trained network weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = args.method
    
    logger.info(f"Starting Multi-Scene Evaluation | Method: {method}")

    # Load the model
    if method == "TRAIN":
        matcher = load_trained_lightglu3d(args.checkpoint, device)
    elif method == "ADAPT":
        matcher = load_trained_adapt(args.checkpoint, device)

    # Load test scene list
    with open(args.scene_list, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]

    overall_precisions = []
    overall_recalls = []
    # Trackers for matchability (sigma)
    all_sigma0 = []
    all_sigma1 = []

    # Process each scene
    for scene in scenes:
        logger.info(f"Processing Scene: {scene}...")
        
        # Load query names
        query_names_file = args.query_dir / scene / "query_image_names_clean.txt"
        with open(query_names_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        pair_dict = load_similar_pairs(args.covisibility_dir / scene / "most_similar_pairs.txt")

        sfm_model_path = args.sfm_dir / scene / "sfm_superpoint+lightglue"
        if not sfm_model_path.exists():
            logger.warning(f"SfM model not found for {scene}. Skipping...")
            continue
            
        query_cams = load_query_cams(args.query_dir / scene / "query_image_cameras.txt")
        
        # Load covisibility
        with open(args.covisibility_dir / scene / "covisibility_results.pkl", "rb") as f:
            covis_dict = pickle.load(f)
 
        scene_precisions = []
        scene_recalls = []

        q_feats_path = args.sfm_dir / scene / "feats-superpoint-n2048.h5"
        p3d_feats_path = args.covisibility_dir / scene / "points3D_feats_cache.h5"
        
        with h5py.File(q_feats_path, "r") as q_feats_h5, h5py.File(p3d_feats_path, "r") as p3d_feats_h5:
            
            # Iterate through all queries in the scene
            for query_name in tqdm(queries, desc=f"Evaluating Queries in {scene}"):
                
                ref_name = pair_dict.get(query_name)
                if not ref_name:
                    continue 
                
                if query_name not in covis_dict:
                    continue
                visible_p3d = covis_dict[query_name]["unique_points"]
                if len(visible_p3d) == 0:
                    continue

                # Get query features
                if query_name not in q_feats_h5:
                    continue
                q_kpts = q_feats_h5[query_name]["keypoints"][:]
                q_desc = q_feats_h5[query_name]["descriptors"][:]

                # Get 3D features
                p3d_desc, p3d_kpts = [], []
                for pid in visible_p3d:
                    pid_str = str(pid)
                    if pid_str in p3d_feats_h5:
                        p3d_desc.append(p3d_feats_h5[pid_str]["descriptors"][:].reshape(256))
                        p3d_kpts.append(p3d_feats_h5[pid_str]["keypoints"][:].reshape(3))
                        
                if len(p3d_kpts) == 0:
                    continue
                    
                p3d_desc = np.vstack(p3d_desc).T 
                p3d_kpts = np.vstack(p3d_kpts)   

                # Compute ground truth
                q_camera = query_cams[query_name]
                q_img_size = np.array([q_camera["intrinsics"]["width"], q_camera["intrinsics"]["height"]])
                
                depth_file = args.depth_dir / scene / f"{Path(query_name).stem}.h5"
                if not depth_file.exists():
                    continue
                with h5py.File(depth_file, 'r') as f_depth:
                    depth_map = f_depth['depth'][:]
                    
                gt_matches0, _ = compute_ground_truth_matches_soft(
                    {"keypoints": q_kpts}, {"keypoints": p3d_kpts}, q_camera, depth_map
                )

                # Predict matches using the pre-loaded neural network
                pred_matches0, score_matrix = compute_trained_lightglu3d(
                    matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device
                )

                # Show sigma
                dustbin_scores0 = score_matrix[:-1, -1] # Unmatched confidence for 2D
                dustbin_scores1 = score_matrix[-1, :-1] # Unmatched confidence for 3D
                # Reverse the math to get Sigma [0, 1]
                sigma0 = 1.0 - np.exp(dustbin_scores0)
                sigma1 = 1.0 - np.exp(dustbin_scores1)
                all_sigma0.append(sigma0)
                all_sigma1.append(sigma1)
                
                # Compute precision and recall
                precision, recall, _, _, _ = compute_precision_recall(pred_matches0, gt_matches0)
                if precision is not None:
                    scene_precisions.append(precision)
                if recall is not None:
                    scene_recalls.append(recall)

        # Scene Summary
        avg_scene_precision = np.mean(scene_precisions) * 100 if scene_precisions else 0
        avg_scene_recall = np.mean(scene_recalls) * 100 if scene_recalls else 0
        
        logger.info(f"Results for Scene: {scene}")
        logger.info(f"Evaluated Queries: {len(scene_precisions)}")
        logger.info(f"Average Precision: {avg_scene_precision:.4f}")
        logger.info(f"Average Recall:    {avg_scene_recall:.4f}")

        overall_precisions.extend(scene_precisions)
        overall_recalls.extend(scene_recalls)

    # Grand Total Summary
    logger.info("="*40)
    logger.info(f"OVERALL {method} RESULTS ")
    logger.info("="*40)
    logger.info(f"Total Scenes Evaluated:  {len(scenes)}")
    logger.info(f"Total Queries Evaluated: {len(overall_precisions)}")
    
    if len(overall_precisions) > 0:
        logger.info(f"Average Precision: {np.mean(overall_precisions):.4f}")
        logger.info(f"Average Recall:    {np.mean(overall_recalls):.4f}")
    else:
        logger.error("No valid queries were evaluated. Please check your data paths.")
    logger.info("="*40)

    show_results_with_sigma(all_sigma0, all_sigma1, num_queries=len(overall_precisions), prefix="test")

if __name__ == "__main__":
    main()