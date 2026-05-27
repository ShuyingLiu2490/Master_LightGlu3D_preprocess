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
from baseline.trained_matcher import load_trained_lightglu3d, load_trained_adapt, compute_trained_lightglu3d
from baseline.rr_baseline import load_similar_pairs, compute_rr_baseline, compute_precision_recall
from baseline.rn_baseline import compute_rn_baseline
from baseline.pr_baseline import compute_pr_baseline
from baseline.pr_baseline_change import compute_pr_baseline_change
from ground_truth.generate_ref_gt_pairs_from_hloc_aachen import compute_ground_truth_matches_aachen
from ground_truth.generate_gt_pairs_by_scene import compute_ground_truth_matches, load_query_cams
from evaluation.pose_estimation_aachen import load_reference_poses, parse_aachen_cameras

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate Precision/Recall Threshold Statistics for All Baselines")
    parser.add_argument('--dataset_type', type=str, default='aachen', choices=['aachen', 'megadepth'])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--covisibility_dir', type=Path, required=True)
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True, help="Directory to save the plotted figure")
    parser.add_argument('--checkpoint_train', type=str, required=True, help="Path to LightGlu3D weights")
    parser.add_argument('--checkpoint_adapt', type=str, required=True, help="Path to LightGlue_Adapt weights")
    parser.add_argument('--hloc_reference', type=Path, help="Required for Aachen GT poses")
    parser.add_argument('--query_dir', type=Path, help="Required for MegaDepth custom query locations")
    parser.add_argument('--scene_list', type=Path, help="Required for MegaDepth scene loops")
    parser.add_argument('--depth_dir', type=Path, help="Required for MegaDepth depth maps")
    args = parser.parse_args()

    args.outputs.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Initializing all Matchers...")
    
    # Initialize matchers
    matchers = {
        "NN": None, 
        "TRAIN": load_trained_lightglu3d(args.checkpoint_train, filter_threshold=0.1, device=device),
        "ADAPT": load_trained_adapt(args.checkpoint_adapt, device),
        "BASELINE": LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    }
    
    methods_to_plot = ["NN", "RR", "RN", "PR", "PRC", "TRAIN", "ADAPT"]

    # Data loaders Setup
    logger.info(f"Loading {args.dataset_type.upper()} dataset structures...")
    if args.dataset_type == "megadepth":
        if not args.query_dir or not args.scene_list or not args.depth_dir:
            raise ValueError("--query_dir, --scene_list, and --depth_dir are strictly required for MegaDepth.")
        with open(args.scene_list, 'r') as f:
            scenes = [line.strip() for line in f if line.strip()]
    else:
        scenes = [""] 

    # Define thresholds (1px to 20px)
    thresholds = np.arange(1.0, 21.0, 1.0)
    global_metrics = {m: {'precision': np.zeros(len(thresholds)), 'recall': np.zeros(len(thresholds))} for m in methods_to_plot}
    evaluated_queries = 0

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

        if args.dataset_type == "aachen":
            aachen_cams = parse_aachen_cameras(scene_query)
            gt_poses = load_reference_poses(args.hloc_reference)
        else:
            query_cams = load_query_cams(scene_query / "query_image_cameras.txt")
            gt_poses = query_cams 

        q_feats_path = scene_sfm / "feats-superpoint-n2048.h5"
        p3d_feats_path = scene_covis / "points3D_feats_cache.h5"
        
        with h5py.File(q_feats_path, "r") as q_feats_h5, h5py.File(p3d_feats_path, "r") as p3d_feats_h5:
            
            desc_label = f"Evaluating Thresholds {scene}" if scene else "Evaluating Thresholds"
            for full_query_name in tqdm(covis_dict.keys(), desc=desc_label):
                base_name = Path(full_query_name).name
                
                # Format camera correctly: keep raw object for GT, and dict for PRC
                camera_dict, raw_camera, gt_pose = None, None, None
                if args.dataset_type == "aachen":
                    if base_name not in gt_poses or full_query_name not in aachen_cams: continue
                    
                    raw_camera = aachen_cams[full_query_name] # Pure PyCOLMAP camera object
                    gt_pose = gt_poses[base_name]             # Dictionary with qvec, tvec
                    
                    # Dictionary format for the PRC
                    camera_dict = {
                        "qvec": gt_pose["qvec"],
                        "tvec": gt_pose["tvec"],
                        "intrinsics": {
                            "model": getattr(raw_camera, 'model_name', getattr(raw_camera.model, 'name', str(raw_camera.model))),
                            "width": raw_camera.width,
                            "height": raw_camera.height,
                            "params": raw_camera.params
                        }
                    }
                else:
                    if full_query_name not in query_cams: continue
                    camera_dict = query_cams[full_query_name]  # Dictionary camera format
                    raw_camera = camera_dict
                    gt_pose = None

                visible_p3d = covis_dict[full_query_name]["unique_points"]
                if len(visible_p3d) == 0 or full_query_name not in q_feats_h5:
                    continue

                q_kpts = q_feats_h5[full_query_name]["keypoints"][:]
                q_desc = q_feats_h5[full_query_name]["descriptors"][:]
                q_img_size = np.array(q_feats_h5[full_query_name]["image_size"][:])

                # Get 3D features and XYZ coordinates
                p3d_desc, p3d_xyz = [], []
                for pid in visible_p3d:
                    pid_str = str(pid)
                    if pid_str in p3d_feats_h5 and int(pid) in sfm_points3D:
                        p3d_desc.append(p3d_feats_h5[pid_str]["descriptors"][:].reshape(256))
                        p3d_xyz.append(sfm_points3D[int(pid)].xyz)
                        
                if len(p3d_xyz) == 0: continue
                p3d_desc = np.vstack(p3d_desc).T 
                p3d_xyz = np.vstack(p3d_xyz)   

                # Prepare Reference Data for RR, RN, PR, PRC
                ref_pose_matrix, ref_cam_obj = None, None
                ref_name = pair_dict.get(full_query_name)
                if ref_name:
                    ref_image_obj = next((img for img in sfm_images.values() if img.name == ref_name), None)
                    if ref_image_obj:
                        ref_R = qvec2rotmat(ref_image_obj.qvec)
                        ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))
                        ref_cam_obj = sfm_cameras[ref_image_obj.camera_id]

                depth_map = None
                if args.dataset_type == "megadepth":
                    depth_file = scene_depth / f"{Path(full_query_name).stem}.h5"
                    if not depth_file.exists(): continue
                    with h5py.File(depth_file, 'r') as f_depth:
                        depth_map = f_depth['depth'][:]

                # Pre-check for the loosest threshold
                if args.dataset_type == "aachen":
                    loose_gt0, _ = compute_ground_truth_matches_aachen(q_kpts, p3d_xyz, raw_camera, gt_pose, pos_reproj_thresh=20.0, neg_reproj_thresh=20.0)
                else:
                    loose_gt0, _ = compute_ground_truth_matches({"keypoints": q_kpts}, {"keypoints": p3d_xyz}, camera_dict, depth_map=depth_map, pos_reproj_thresh=20.0, neg_reproj_thresh=20.0)
                if np.sum(loose_gt0 >= 0) == 0:
                    continue
                evaluated_queries += 1

                # Predict matches for 7 methods
                preds = {}
                preds["NN"] = compute_nn_baseline(q_desc, p3d_desc, device)
                preds["TRAIN"] = compute_trained_lightglu3d(matchers['TRAIN'], q_kpts, q_desc, q_img_size, p3d_xyz, p3d_desc, device)
                preds["ADAPT"] = compute_trained_lightglu3d(matchers['ADAPT'], q_kpts, q_desc, q_img_size, p3d_xyz, p3d_desc, device)

                # Baselines
                preds["RR"], _, _, _, _ = compute_rr_baseline(matchers['BASELINE'], q_kpts, q_desc, q_img_size, p3d_xyz, p3d_desc, ref_pose_matrix, device)
                preds["RN"], _, _, _, _ = compute_rn_baseline(matchers['BASELINE'], q_kpts, q_desc, q_img_size, p3d_xyz, p3d_desc, ref_pose_matrix, device)
                preds["PR"], _, _, _, _ = compute_pr_baseline(matchers['BASELINE'], q_kpts, q_desc, q_img_size, p3d_xyz, p3d_desc, ref_pose_matrix, ref_cam_obj, device)
                preds["PRC"], _, _, _, _ = compute_pr_baseline_change(matchers['BASELINE'], q_kpts, q_desc, q_img_size, p3d_xyz, p3d_desc, ref_pose_matrix, camera_dict, device)

                # Iterate through exact thresholds dynamically
                for t_idx, t_val in enumerate(thresholds):
                    
                    if args.dataset_type == "aachen":
                        gt0, _ = compute_ground_truth_matches_aachen(q_kpts, p3d_xyz, raw_camera, gt_pose, pos_reproj_thresh=t_val, neg_reproj_thresh=t_val)
                    else:
                        gt0, _ = compute_ground_truth_matches({"keypoints": q_kpts}, {"keypoints": p3d_xyz}, camera_dict, depth_map=depth_map, pos_reproj_thresh=t_val, neg_reproj_thresh=t_val)
                        
                    for m_name, pred_array in preds.items():
                        precision, recall, _, _, _ = compute_precision_recall(pred_array, gt0)
                        global_metrics[m_name]['precision'][t_idx] += precision
                        global_metrics[m_name]['recall'][t_idx] += recall

    # Average the metrics
    if evaluated_queries == 0:
        logger.error("No valid queries evaluated! Please check your file paths.")
        return
        
    for m in methods_to_plot:
        global_metrics[m]['precision'] /= evaluated_queries
        global_metrics[m]['recall'] /= evaluated_queries

    # Plotting
    logger.info(f"Generating Plot for {evaluated_queries} Queries...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Only show 3 matchers
    colors = {
        "NN": "#e74c3c",    # Red
        # "RR": "#f569e2",    # Pink
        # "RN": "#8e44ad",    # Purple
        # "PR": "#f1c40f",    # Yellow
        "PRC": "#e67e22",   # Orange
        "TRAIN": "#2ecc71" # Green
        # "ADAPT": "#3498db"  # Blue
    }
    markers = {
        "NN": "s", 
        # "RR": "v", 
        # "RN": "^", 
        # "PR": "<", 
        "PRC": ">", 
        "TRAIN": "o"
        # "ADAPT": "D"
    }

    # Subplot 1: Precision
    for m in methods_to_plot:
        ax1.plot(thresholds, global_metrics[m]['precision'], label=m, color=colors[m], marker=markers[m], linewidth=2.5, markersize=8)
    ax1.set_title("Match Precision", fontsize=15, fontweight='bold')
    ax1.set_xlabel("Pixel Threshold (px)", fontsize=13)
    ax1.set_ylabel("Match Precision", fontsize=13)
    ax1.set_xticks(thresholds)
    ax1.legend(fontsize=11)

    # Subplot 2: Recall
    for m in methods_to_plot:
        ax2.plot(thresholds, global_metrics[m]['recall'], label=m, color=colors[m], marker=markers[m], linewidth=2.5, markersize=8)
    ax2.set_title("Match Recall", fontsize=15, fontweight='bold')
    ax2.set_xlabel("Pixel Threshold (px)", fontsize=13)
    ax2.set_ylabel("Match Recall", fontsize=13)
    ax2.set_xticks(thresholds)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    
    # Save Output
    plot_path = args.outputs / f"{args.dataset_type}_threshold_statistics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Statistics Figure successfully saved to: {plot_path}")

if __name__ == "__main__":
    main()