import argparse
import logging
import numpy as np
import torch
import time
from pathlib import Path
from tqdm import tqdm
from lightglue import LightGlue
from baselines_and_trained_matcher.mnn_baseline import compute_nn_baseline
from baselines_and_trained_matcher.pr_lg_baseline import compute_pr_baseline
from baselines_and_trained_matcher.hloc_fair_baseline import compute_hloc_baseline
from baselines_and_trained_matcher.trained_matcher import (
    load_trained_lightglu3d, 
    compute_trained_lightglu3d, 
    compute_trained_lightglu3d_greedy_dynamic
)
from .calculate_absolute_pose import log_metrics, evaluate_pose, estimate_pose_blind
from preprocess_extra.dataloader import CambridgeLoader, AachenLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_inference_loop(loader, matchers, args, device, is_cambridge=False):
    t_errors, r_errors = [], []
    failed_pnp_count = 0
    estimated_poses = {}
    
    total_match_time = 0.0
    total_pnp_time = 0.0
    valid_queries = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for data in tqdm(loader, desc="Evaluating Queries"):
        if not data["is_valid"]:
            failed_pnp_count += 1
            if is_cambridge:
                t_errors.append(np.inf); r_errors.append(np.inf)
            continue
            
        valid_queries += 1

        # GPU Timing: Matching
        start_event.record()

        if args.method == "HLOC":
            pred_matches0 = compute_hloc_baseline(matchers['baseline'], data["q_kpts"], data["q_desc"], data["q_img_size"], data["ref_data_list"], data["p3d_indices_map"], device)
        elif args.method == "MNN":
            pred_matches0 = compute_nn_baseline(data["q_desc"], data["p3d_desc"], device)
        elif args.method == "PR":
            pred_matches0, _, _, _, _ = compute_pr_baseline(matchers['baseline'], data["q_kpts"], data["q_desc"], data["q_img_size"], data["p3d_kpts"], data["p3d_desc"], data["ref_pose_matrix"], data["camera_dict"], device)
        elif args.method == "TRAIN":
            if not is_cambridge and args.greedy_or_not:
                pred_matches0, _ = compute_trained_lightglu3d_greedy_dynamic(matchers['lightglu3d'], data["q_kpts"], data["q_desc"], data["q_img_size"], data["p3d_kpts"], data["p3d_desc"], device, min_matches=args.min_matches)
            else:
                pred_matches0, _ = compute_trained_lightglu3d(matchers['lightglu3d'], data["q_kpts"], data["q_desc"], data["q_img_size"], data["p3d_kpts"], data["p3d_desc"], device)

        end_event.record()
        torch.cuda.synchronize()
        total_match_time += start_event.elapsed_time(end_event)

        valid_mask = pred_matches0 > -1
        matched_2d = data["q_kpts"][valid_mask] + 0.5
        matched_3d = data["p3d_for_pnp"][pred_matches0[valid_mask]] 

        # CPU Timing: POSE ESTIMATION
        t0 = time.perf_counter()
        
        if is_cambridge:
            pose_res = evaluate_pose(matched_2d, matched_3d, data["camera_dict"], data["q_img_size"], args.max_error)
        else:
            pose_res = estimate_pose_blind(matched_2d, matched_3d, data["colmap_cam"], data["q_img_size"], args.max_error)
            
        t1 = time.perf_counter()
        total_pnp_time += (t1 - t0) * 1000  # Convert seconds to ms

        # Handle Results
        if pose_res is not None:
            if is_cambridge:
                t_errors.append(pose_res["t_error"])
                r_errors.append(pose_res["r_error_deg"])
            else:
                estimated_poses[data["query_name"]] = pose_res
        else:
            failed_pnp_count += 1
            if is_cambridge:
                t_errors.append(np.inf); r_errors.append(np.inf)

    return t_errors, r_errors, estimated_poses, failed_pnp_count, valid_queries, total_match_time, total_pnp_time

def main():
    parser = argparse.ArgumentParser(description="Unified Pose Estimation for Cambridge and Aachen")
    parser.add_argument('--dataset_type', type=str, required=True, choices=['cambridge', 'aachen'])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--covisibility_dir', type=Path, required=True)
    parser.add_argument('--query_dir', type=Path, required=True)
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--scene_list', type=Path, default=None)
    parser.add_argument('--method', type=str, required=True, choices=['MNN', 'PR', 'TRAIN', 'HLOC'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--max_error', type=float, default=12.0)
    parser.add_argument('--filter_threshold', type=float, default=0.1)
    parser.add_argument('--outputs', type=Path, default=None)
    parser.add_argument('--greedy_or_not', type=bool, default=False)
    parser.add_argument('--min_matches', type=int, default=800)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    matchers = {}
    if args.method in ["PR", "HLOC"]:
        matchers['baseline'] = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    elif args.method == "TRAIN":
        if args.checkpoint is None: raise ValueError("--checkpoint required for TRAIN")
        matchers['lightglu3d'] = load_trained_lightglu3d(args.checkpoint, device, filter_threshold=args.filter_threshold)

    if args.dataset_type == 'cambridge':
        if not args.scene_list: raise ValueError("--scene_list required for Cambridge")
        with open(args.scene_list, 'r') as f:
            scenes = [line.strip() for line in f if line.strip()]
            
        all_t_errors, all_r_errors = [], []
        global_failed, global_total = 0, 0
        global_match_time, global_pnp_time = 0.0, 0.0

        for scene in scenes:
            loader = CambridgeLoader(scene, args)
            if not getattr(loader, 'is_valid', False): continue
                
            t_errs, r_errs, _, fails, total_valid, m_time, p_time = run_inference_loop(loader, matchers, args, device, is_cambridge=True)
            
            if len(loader) > 0:
                log_metrics(t_errs, r_errs, fails, len(loader), method_label=f"{args.method} - SCENE: {scene}")
                
            all_t_errors.extend(t_errs)
            all_r_errors.extend(r_errs)
            global_failed += fails
            global_total += total_valid
            global_match_time += m_time
            global_pnp_time += p_time

        if len(all_t_errors) > 0:
            log_metrics(all_t_errors, all_r_errors, global_failed, len(all_t_errors), method_label=f"{args.method} - CAMBRIDGE SUMMARY")
            logger.info(f"Avg Matching Time: {global_match_time / global_total:.2f} ms/query")
            logger.info(f"Avg PnP Time:      {global_pnp_time / global_total:.2f} ms/query")

    elif args.dataset_type == 'aachen':
        if not args.outputs: raise ValueError("--outputs directory required for Aachen")
        
        loader = AachenLoader(args)
        _, _, estimated_poses, failed, total_valid, m_time, p_time = run_inference_loop(loader, matchers, args, device, is_cambridge=False)

        logger.info("="*40)
        logger.info(f"Aachen Evaluation Completed: {args.method}")
        logger.info(f"Total Queries:      {len(loader)}")
        logger.info(f"Successfully PnP:   {len(estimated_poses)}")
        logger.info(f"Failed PnP:         {failed}")
        logger.info(f"Avg Matching Time:  {m_time / total_valid:.2f} ms/query" if total_valid > 0 else "Avg Matching Time: N/A")
        logger.info(f"Avg PnP Time:       {p_time / total_valid:.2f} ms/query" if total_valid > 0 else "Avg PnP Time: N/A")
        logger.info("="*40)

        output_file = args.outputs / f"Aachen_v1_1_eval_{args.method}.txt"
        with open(output_file, 'w') as f:
            for img_name, pose in estimated_poses.items():
                q, t = pose['qvec'], pose['tvec']
                f.write(f"{Path(img_name).name} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]}\n")

if __name__ == "__main__":
    main()