import argparse
import logging
import pickle
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat
from lightglue import LightGlue
from baseline.nn_baseline import compute_nn_baseline
from baseline.trained_matcher import load_trained_lightglu3d, compute_trained_lightglu3d_greedy_dynamic, compute_trained_lightglu3d_dynamic
from baseline.rr_baseline import load_similar_pairs
from baseline.pr_baseline_change import compute_pr_baseline_change
from ground_truth.generate_ref_gt_pairs_from_hloc_aachen import compute_ground_truth_matches_aachen
from ground_truth.generate_gt_pairs_by_scene import compute_ground_truth_matches, load_query_cams
from evaluation.pose_estimation_aachen import load_reference_poses, parse_aachen_cameras

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate Absolute Match Statistics by Original Query Index")
    parser.add_argument('--dataset_type', type=str, default='aachen', choices=['aachen', 'megadepth'])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--covisibility_dir', type=Path, required=True)
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True, help="Directory to save the plotted figure")
    parser.add_argument('--checkpoint_train', type=str, required=True, help="Path to LightGlu3D weights")
    parser.add_argument('--hloc_reference', type=Path, help="Required for Aachen GT poses")
    parser.add_argument('--query_dir', type=Path, help="Required for MegaDepth custom query locations")
    parser.add_argument('--scene_list', type=Path, help="Required for MegaDepth scene loops")
    parser.add_argument('--depth_dir', type=Path, help="Required for MegaDepth depth maps")
    parser.add_argument('--filter_threshold', type=float, default=0.1, help="Filter threshold for TRAIN LightGlu3D matcher")
    parser.add_argument('--min_matches', type=int, default=800, help="Minimum matches for dynamic LightGlue")
    args = parser.parse_args()

    stat_dir = args.outputs
    stat_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_log_path = stat_dir / f"{args.dataset_type}_query_results.txt"
    with open(results_log_path, "w") as f_log:
        f_log.write(f"Absolute Match Results for {args.dataset_type.upper()} ...\n")
    logger.info(f"Query statistics will be saved to: {results_log_path}")
    
    logger.info("Initializing NN, PRC, and TRAIN Matchers...")
    
    matchers = {
        "TRAIN": load_trained_lightglu3d(args.checkpoint_train, device, args.filter_threshold),
        "BASELINE": LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    }
    
    methods_to_plot = ["NN", "PRC", "TRAIN"]
    colors = {"NN": "#e74c3c", "PRC": "#e67e22", "TRAIN": "#2ecc71"}
    # Trackers for Thresholds AND Failed Images
    dynamic_threshold_stats = {0.05: 0, 0.025: 0, 0.015: 0, 0.005: 0, 0.001: 0, "Failed": 0}

    logger.info(f"Loading {args.dataset_type.upper()} dataset structures...")
    if args.dataset_type == "megadepth":
        if not args.query_dir or not args.scene_list or not args.depth_dir:
            raise ValueError("--query_dir, --scene_list, and --depth_dir are strictly required for MegaDepth.")
        with open(args.scene_list, 'r') as f:
            scenes = [line.strip() for line in f if line.strip()]
    else:
        scenes = [""] 

    # Trackers for Aachen
    aachen_metrics_day = {m: {'pred': [], 'corr': []} for m in methods_to_plot}
    aachen_metrics_night = {m: {'pred': [], 'corr': []} for m in methods_to_plot}
    aachen_idx_day, aachen_idx_night = [], []

    # Process by scene
    for scene in scenes:
        scene_covis = args.covisibility_dir / scene if scene else args.covisibility_dir
        scene_sfm = args.sfm_dir / scene if scene else args.sfm_dir
        scene_query = args.query_dir / scene if (scene and args.query_dir) else (args.query_dir if args.query_dir else args.dataset / "queries")
        scene_depth = args.depth_dir / scene if (scene and args.depth_dir) else None
        
        covis_path = scene_covis / "covisibility_results.pkl"
        if not covis_path.exists(): continue
            
        with open(covis_path, "rb") as f:
            covis_dict = pickle.load(f)

        sfm_model_path = scene_sfm / "sfm_superpoint+lightglue"
        sfm_cameras, sfm_images, sfm_points3D = rw.read_model(sfm_model_path, ext=".bin")

        pair_path = scene_covis / "most_similar_pairs.txt"
        pair_dict = load_similar_pairs(pair_path) if pair_path.exists() else {}

        # Load exact query lists to determine order 
        day_queries, night_queries, mega_queries = [], [], []
        if args.dataset_type == "aachen":
            aachen_cams = parse_aachen_cameras(scene_query)
            gt_poses = load_reference_poses(args.hloc_reference)
            day_list_path = scene_query / "day_time_queries_with_intrinsics.txt"
            night_list_path = scene_query / "night_time_queries_with_intrinsics.txt"
            if day_list_path.exists():
                with open(day_list_path, 'r') as f:
                    day_queries = [line.strip().split()[0] for line in f if line.strip() and not line.startswith("#")]
            if night_list_path.exists():
                with open(night_list_path, 'r') as f:
                    night_queries = [line.strip().split()[0] for line in f if line.strip() and not line.startswith("#")]
        else:
            query_cams = load_query_cams(scene_query / "query_image_cameras.txt")
            gt_poses = query_cams 
            query_list_path = scene_query / "query_image_names_clean.txt"
            if query_list_path.exists():
                with open(query_list_path, 'r') as f:
                    mega_queries = [line.strip() for line in f if line.strip()]

        q_feats_path = scene_sfm / "feats-superpoint-n2048.h5"
        p3d_feats_path = scene_covis / "points3D_feats_cache.h5"
        
        mega_scene_metrics = {m: {'pred': [], 'corr': []} for m in methods_to_plot}
        mega_scene_idx = []

        with h5py.File(q_feats_path, "r") as q_feats_h5, h5py.File(p3d_feats_path, "r") as p3d_feats_h5:
            
            queries_to_eval = day_queries + night_queries if args.dataset_type == "aachen" else mega_queries
            
            desc_label = f"Evaluating Matches {scene}" if scene else "Evaluating Matches"
            for full_query_name in tqdm(queries_to_eval, desc=desc_label, leave=False):
                base_name = Path(full_query_name).name
                
                category = None
                q_index = -1
                
                if args.dataset_type == "aachen":
                    if full_query_name in day_queries:
                        category = "day"
                        q_index = day_queries.index(full_query_name)
                    elif full_query_name in night_queries:
                        category = "night"
                        q_index = night_queries.index(full_query_name)
                    else: continue
                else:
                    category = "megadepth"
                    if full_query_name in mega_queries:
                        q_index = mega_queries.index(full_query_name)
                    else: continue

                # Default variables for this iteration
                query_pred_counts = {m: 0 for m in methods_to_plot}
                query_correct_counts = {m: 0 for m in methods_to_plot}
                used_th_str = "N/A"
                can_run_matcher = True

                camera_dict, raw_camera, gt_pose = None, None, None
                if args.dataset_type == "aachen":
                    if base_name not in gt_poses or full_query_name not in aachen_cams: 
                        can_run_matcher = False
                    else:
                        raw_camera = aachen_cams[full_query_name]
                        gt_pose = gt_poses[base_name] 
                        camera_dict = {
                            "qvec": gt_pose["qvec"], "tvec": gt_pose["tvec"],
                            "intrinsics": {
                                "model": getattr(raw_camera, 'model_name', getattr(raw_camera.model, 'name', str(raw_camera.model))),
                                "width": raw_camera.width, "height": raw_camera.height, "params": raw_camera.params
                            }
                        }
                else:
                    if full_query_name not in query_cams: 
                        can_run_matcher = False
                    else:
                        camera_dict = query_cams[full_query_name]
                        raw_camera = camera_dict
                        gt_pose = None

                if full_query_name not in covis_dict: can_run_matcher = False
                else:
                    visible_p3d = covis_dict[full_query_name]["unique_points"]
                    if len(visible_p3d) == 0: can_run_matcher = False

                if full_query_name not in q_feats_h5: can_run_matcher = False

                if can_run_matcher:
                    q_kpts = q_feats_h5[full_query_name]["keypoints"][:]
                    q_desc = q_feats_h5[full_query_name]["descriptors"][:]
                    q_img_size = np.array(q_feats_h5[full_query_name]["image_size"][:])

                    p3d_desc, p3d_xyz = [], []
                    for pid in visible_p3d:
                        pid_str = str(pid)
                        if pid_str in p3d_feats_h5 and int(pid) in sfm_points3D:
                            p3d_desc.append(p3d_feats_h5[pid_str]["descriptors"][:].reshape(256))
                            p3d_xyz.append(sfm_points3D[int(pid)].xyz)
                            
                    if len(p3d_xyz) == 0: 
                        can_run_matcher = False
                    else:
                        p3d_desc = np.vstack(p3d_desc).T 
                        p3d_xyz = np.vstack(p3d_xyz)   

                        ref_pose_matrix = None
                        ref_name = pair_dict.get(full_query_name)
                        if ref_name:
                            ref_image_obj = next((img for img in sfm_images.values() if img.name == ref_name), None)
                            if ref_image_obj:
                                ref_R = qvec2rotmat(ref_image_obj.qvec)
                                ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))

                        depth_map = None
                        if args.dataset_type == "megadepth":
                            depth_file = scene_depth / f"{Path(full_query_name).stem}.h5"
                            if depth_file.exists():
                                with h5py.File(depth_file, 'r') as f_depth:
                                    depth_map = f_depth['depth'][:]

                        # Run matchers
                        preds = {}
                        preds["NN"] = compute_nn_baseline(q_desc, p3d_desc, device)
                        preds["TRAIN"], _, used_th = compute_trained_lightglu3d_dynamic(matchers['TRAIN'], q_kpts, q_desc, q_img_size, p3d_xyz, p3d_desc, device, dynamic_threshold_stats, min_matches=args.min_matches)
                        used_th_str = f"{used_th:.3f}"
                        preds["PRC"], _, _, _, _ = compute_pr_baseline_change(matchers['BASELINE'], q_kpts, q_desc, q_img_size, p3d_xyz, p3d_desc, ref_pose_matrix, camera_dict, device)

                        # Check ground truth
                        has_gt = False
                        if args.dataset_type == "aachen":
                            gt0, _ = compute_ground_truth_matches_aachen(q_kpts, p3d_xyz, raw_camera, gt_pose)
                        else:
                            gt0, _ = compute_ground_truth_matches({"keypoints": q_kpts}, {"keypoints": p3d_xyz}, camera_dict, depth_map=depth_map)
                        if np.sum(gt0 >= 0) > 0:
                            has_gt = True

                        # Count matches
                        for m_name in methods_to_plot:
                            pred_array = preds[m_name]
                            valid_mask = pred_array > -1
                            query_pred_counts[m_name] = np.sum(valid_mask)
                            
                            # If GT exists, calculate correct matches, otherwise it defaults to 0
                            if has_gt:
                                correct_mask = (pred_array == gt0) & valid_mask & (gt0 >= 0)
                                query_correct_counts[m_name] = np.sum(correct_mask)

                else:
                    dynamic_threshold_stats["Failed"] += 1

                # Append metrics for plotting
                if args.dataset_type == "aachen":
                    if category == "day": 
                        aachen_idx_day.append(q_index)
                        for m_name in methods_to_plot:
                            aachen_metrics_day[m_name]['pred'].append(query_pred_counts[m_name])
                            aachen_metrics_day[m_name]['corr'].append(query_correct_counts[m_name])
                    else: 
                        aachen_idx_night.append(q_index)
                        for m_name in methods_to_plot:
                            aachen_metrics_night[m_name]['pred'].append(query_pred_counts[m_name])
                            aachen_metrics_night[m_name]['corr'].append(query_correct_counts[m_name])
                else:
                    mega_scene_idx.append(q_index)
                    for m_name in methods_to_plot:
                        mega_scene_metrics[m_name]['pred'].append(query_pred_counts[m_name])
                        mega_scene_metrics[m_name]['corr'].append(query_correct_counts[m_name])

                # Save in log unconditionally
                log_line = (f"Query: {full_query_name:<30} | Thresh: {used_th_str:>5} | "
                            f"NN (P/C): {query_pred_counts['NN']:>4}/{query_correct_counts['NN']:<4} | "
                            f"PRC (P/C): {query_pred_counts['PRC']:>4}/{query_correct_counts['PRC']:<4} | "
                            f"TRAIN (P/C): {query_pred_counts['TRAIN']:>4}/{query_correct_counts['TRAIN']:<4}\n")
                
                with open(results_log_path, "a") as f_log:
                    f_log.write(log_line)


        # Plot MegaDepth
        if args.dataset_type == "megadepth" and mega_scene_idx:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
            
            # Sort strictly by the original query index
            sorted_idx = np.argsort(mega_scene_idx)
            x_vals = np.array(mega_scene_idx)[sorted_idx]
            
            for m in methods_to_plot:
                y_pred = np.array(mega_scene_metrics[m]['pred'])[sorted_idx]
                y_corr = np.array(mega_scene_metrics[m]['corr'])[sorted_idx]
                
                # Plot predicted matches
                ax1.plot(x_vals, y_pred, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=4)
                # Plot correct matches
                ax2.plot(x_vals, y_corr, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=4)
            
            ax1.set_title(f"MegaDepth Scene {scene} - Predicted Matches", fontsize=15, fontweight='bold')
            ax1.set_ylabel("Predicted Matches", fontsize=13)
            ax1.legend(fontsize=11)
            
            ax2.set_title(f"MegaDepth Scene {scene} - Correct Matches", fontsize=15, fontweight='bold')
            ax2.set_xlabel("Original Query Index", fontsize=13)
            ax2.set_ylabel("Correct Matches", fontsize=13)
            ax2.legend(fontsize=11)
            
            plt.tight_layout()
            plot_path = stat_dir / f"megadepth_scene_{scene}_absolute_matches_ordered.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved MegaDepth Scene plot to: {plot_path}")

    # Plot Aachen
    if args.dataset_type == "aachen":
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Day queries
        if aachen_idx_day:
            sorted_idx = np.argsort(aachen_idx_day)
            x_vals = np.array(aachen_idx_day)[sorted_idx]
            for m in methods_to_plot:
                y_pred = np.array(aachen_metrics_day[m]['pred'])[sorted_idx]
                y_corr = np.array(aachen_metrics_day[m]['corr'])[sorted_idx]
                ax1.plot(x_vals, y_pred, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=4)
                ax2.plot(x_vals, y_corr, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=4)
                
        ax1.set_title("Aachen Day - Predicted Matches", fontsize=15, fontweight='bold')
        ax1.set_ylabel("Predicted Matches", fontsize=13)
        ax1.legend(fontsize=11)
        
        ax2.set_title("Aachen Day - Correct Matches", fontsize=15, fontweight='bold')
        ax2.set_ylabel("Correct Matches", fontsize=13)
        ax2.legend(fontsize=11)
        
        # Night queries
        if aachen_idx_night:
            sorted_idx = np.argsort(aachen_idx_night)
            x_vals = np.array(aachen_idx_night)[sorted_idx]
            for m in methods_to_plot:
                y_pred = np.array(aachen_metrics_night[m]['pred'])[sorted_idx]
                y_corr = np.array(aachen_metrics_night[m]['corr'])[sorted_idx]
                ax3.plot(x_vals, y_pred, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=4)
                ax4.plot(x_vals, y_corr, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=4)
                
        ax3.set_title("Aachen Night - Predicted Matches", fontsize=15, fontweight='bold')
        ax3.set_xlabel("Original Query Index", fontsize=13)
        ax3.set_ylabel("Predicted Matches", fontsize=13)
        ax3.legend(fontsize=11)

        ax4.set_title("Aachen Night - Correct Matches", fontsize=15, fontweight='bold')
        ax4.set_xlabel("Original Query Index", fontsize=13)
        ax4.set_ylabel("Correct Matches", fontsize=13)
        ax4.legend(fontsize=11)
        
        plt.tight_layout()
        plot_path = stat_dir / "aachen_absolute_matches_ordered.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Aachen plot to: {plot_path}")

    # Dynamic threshold statistics
    logger.info("="*50)
    logger.info("Dynamic Threshold Statistics")
    logger.info("="*50)
    for key, count in dynamic_threshold_stats.items():
        if key == "Failed":
            logger.info(f"Skipped/Failed completely for {count} queries.")
        else:
            logger.info(f"Threshold [{key:.3f}] used for {count} queries.")
    logger.info("="*50)

if __name__ == "__main__":
    main()