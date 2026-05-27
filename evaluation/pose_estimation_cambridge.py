# For baseline: NN(Nearest Neighbour), RR(Rotate+Remove_coord), 
#               RN(Rotate+Normalize), PR(Project to Reference),
#               HLOC(2D-2D LightGlue + Lift to 3D)  
# For train: TRAIN(Lightglu3d two self and one bidirectional cross), 
#            ADAPT(lightglue+adapter)

# Before use it, add the gluefactory path in the terminal
# export PYTHONPATH="/home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match:$PYTHONPATH"

import argparse
import logging
import torch
from pathlib import Path
from tqdm import tqdm
from lightglue import LightGlue
from visualization.visualize_matches import load_trained_lightglu3d, load_trained_adapt, compute_trained_lightglu3d, get_most_similar_ref
from .pose_estimation import process_scene, log_metrics

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pose Accuracy over Cambridge Landmarks")
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--covisibility_dir', type=Path, required=True)
    parser.add_argument('--query_dir', type=Path, required=True)
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--scene_list', type=Path, required=True, help="Path to txt file with list of Cambridge scenes")
    parser.add_argument('--method', type=str, required=True, choices=['NN', 'RR', 'RN', 'PR', 'TRAIN', 'ADAPT', 'HLOC'], 
                        help="Matching method to evaluate")
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help="Path to trained network weights")
    parser.add_argument('--max_error', type=float, default=3.0, help="RANSAC Reprojection Error Threshold (pixels)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = args.method

    with open(args.scene_list, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]
        
    logger.info(f"Starting Pose Evaluation for {len(scenes)} Cambridge scenes using {method} (RANSAC max_error: {args.max_error}px)")

    # Initialize matchers
    matchers = {}
    if method in ["RR", "RN", "PR", "HLOC"]:
        matchers['baseline'] = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    elif method == "TRAIN":
        if args.checkpoint is None: raise ValueError("--checkpoint must be provided when using the TRAIN method.")
        matchers['lightglu3d'] = load_trained_lightglu3d(args.checkpoint, device)
    elif method == "ADAPT":
        if args.checkpoint is None: raise ValueError("--checkpoint must be provided when using the ADAPT method.")
        matchers['adapt'] = load_trained_adapt(args.checkpoint, device)

    # Global accumulators
    all_t_errors = []
    all_r_errors = []
    global_failed_pnp_count = 0
    global_total_queries = 0

    for scene in scenes:
        logger.info(f"Processing Scene: {scene}...")
        
        t_errs, r_errs, fails, total = process_scene(scene, args, matchers, device, is_megadepth=False)
        
        if total > 0:
            log_metrics(t_errs, r_errs, fails, total, method_label=f"{method} - SCENE: {scene}")
        
        all_t_errors.extend(t_errs)
        all_r_errors.extend(r_errs)
        global_failed_pnp_count += fails
        global_total_queries += total

    if global_total_queries > 0:
        log_metrics(all_t_errors, all_r_errors, global_failed_pnp_count, global_total_queries, 
            method_label=f"{method} - CAMBRIDGE SUMMARY"
        )

if __name__ == "__main__":
    main()