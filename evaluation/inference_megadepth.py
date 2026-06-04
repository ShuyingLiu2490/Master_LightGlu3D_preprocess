import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from baselines_and_trained_matcher.mnn_baseline import compute_nn_baseline
from baselines_and_trained_matcher.pr_lg_baseline import compute_pr_baseline
from baselines_and_trained_matcher.trained_matcher import load_trained_lightglu3d, compute_trained_lightglu3d
from lightglue import LightGlue
from utils.matching_performance import compute_precision_recall
from preprocess_Megadepth.dataloader import MegaDepthLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Precision/Recall for TRAIN and Baselines across scenes")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Undistorted_SfM")
    parser.add_argument('--covisibility_dir', type=Path, required=True, help="Path to covisibility")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to sfm outputs")
    parser.add_argument('--depth_dir', type=Path, required=True, help="Path to depth maps")
    parser.add_argument('--scene_list', type=Path, required=True, help="Path to text file containing list of scenes")
    parser.add_argument('--method', type=str, required=True, choices=['TRAIN', 'MNN', 'PR'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--filter_threshold', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = args.method
    
    logger.info(f"Starting Multi-Scene Evaluation | Method: {method}")

    if method == "TRAIN":
        if args.checkpoint is None: raise ValueError("--checkpoint required for TRAIN")
        matcher = load_trained_lightglu3d(args.checkpoint, device, filter_threshold=args.filter_threshold)
    elif method == "PR":
        matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    elif method == "MNN":
        matcher = None 

    with open(args.scene_list, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]

    overall_precisions, overall_recalls = [], []
    total_matching_time_ms = 0.0

    # Timing Events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for scene in scenes:
        logger.info(f"Processing Scene: {scene}...")
        
        # Instantiate loader
        scene_loader = MegaDepthLoader(scene, args)
        
        if not scene_loader.is_valid:
            continue

        scene_precisions, scene_recalls = [], []

        for data in tqdm(scene_loader, total=len(scene_loader), desc=f"Evaluating Queries in {scene}"):
            
            # Start GPU Timing
            start_event.record()

            if method == "TRAIN":
                pred_matches0, _ = compute_trained_lightglu3d(
                    matcher, data["q_kpts"], data["q_desc"], data["q_img_size"], 
                    data["p3d_kpts"], data["p3d_desc"], device
                )
            elif method == "MNN":
                pred_matches0 = compute_nn_baseline(data["q_desc"], data["p3d_desc"], device)
            elif method == "PR":
                if data["ref_pose_matrix"] is None: continue
                pred_matches0, _, _, _, _ = compute_pr_baseline(
                    matcher, data["q_kpts"], data["q_desc"], data["q_img_size"], 
                    data["p3d_kpts"], data["p3d_desc"], data["ref_pose_matrix"], 
                    data["camera"], device
                )

            end_event.record()
            torch.cuda.synchronize()
            total_matching_time_ms += start_event.elapsed_time(end_event)
            # End GPU Timing

            # Compute precision and recall
            precision, recall, _, _, _ = compute_precision_recall(pred_matches0, data["gt_matches0"])
            if precision is not None: scene_precisions.append(precision)
            if recall is not None: scene_recalls.append(recall)

        overall_precisions.extend(scene_precisions)
        overall_recalls.extend(scene_recalls)

    # Grand Total Summary
    logger.info("="*40)
    logger.info(f"OVERALL {method} RESULTS")
    logger.info("="*40)
    logger.info(f"Total Scenes Evaluated:  {len(scenes)}")
    logger.info(f"Total Queries Evaluated: {len(overall_precisions)}")
    
    if len(overall_precisions) > 0:
        avg_time = total_matching_time_ms / len(overall_precisions)
        logger.info(f"Average Precision:     {np.mean(overall_precisions)*100:.2f}%")
        logger.info(f"Average Recall:        {np.mean(overall_recalls)*100:.2f}%")
        logger.info(f"Avg Matching Time:     {avg_time:.2f} ms / query")
    else:
        logger.error("No valid queries were evaluated. Please check your data paths.")
    logger.info("="*40)

if __name__ == "__main__":
    main()