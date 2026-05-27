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
from baseline.trained_matcher import load_trained_lightglu3d, compute_trained_lightglu3d
from baseline.rr_baseline import load_similar_pairs
from baseline.pr_baseline import compute_pr_baseline
from evaluation.pose_estimation_aachen import load_reference_poses, parse_aachen_cameras
from ground_truth.generate_gt_pairs_by_scene import load_query_cams

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate Reprojection Error Sparkline Statistics")
    parser.add_argument('--dataset_type', type=str, default='aachen', choices=['aachen', 'megadepth'])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--covisibility_dir', type=Path, required=True)
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True, help="Directory to save the plotted figure")
    parser.add_argument('--checkpoint_train', type=str, required=True, help="Path to LightGlu3D weights")
    parser.add_argument('--hloc_reference', type=Path, help="Required for Aachen GT poses")
    parser.add_argument('--query_dir', type=Path, help="Required for MegaDepth custom query locations")
    parser.add_argument('--scene_list', type=Path, help="Required for MegaDepth scene loops")
    args = parser.parse_args()

    args.outputs.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Initializing NN, PR, and TRAIN Matchers...")
    
    matchers = {
        "NN": None, 
        "TRAIN": load_trained_lightglu3d(args.checkpoint_train, device),
        "BASELINE": LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    }
    
    # Initialize 3 methods
    methods_to_plot = ["NN", "PR", "TRAIN"]
    thresholds = [3.0, 6.0, 9.0, 12.0]

    logger.info(f"Loading {args.dataset_type.upper()} dataset structures...")
    if args.dataset_type == "megadepth":
        with open(args.scene_list, 'r') as f:
            scenes = [line.strip() for line in f if line.strip()]
    else:
        scenes = [""] 

    # Trackers for Aachen
    aachen_day = {m: {t: [] for t in thresholds} for m in methods_to_plot}
    aachen_night = {m: {t: [] for t in thresholds} for m in methods_to_plot}
    aachen_idx_day, aachen_idx_night = [], []
    evaluated_queries = 0

    # Process by scene
    for scene in scenes:
        scene_covis = args.covisibility_dir / scene if scene else args.covisibility_dir
        scene_sfm = args.sfm_dir / scene if scene else args.sfm_dir
        scene_query = args.query_dir / scene if (scene and args.query_dir) else (args.query_dir if args.query_dir else args.dataset / "queries")
        
        covis_path = scene_covis / "covisibility_results.pkl"
        if not covis_path.exists(): continue
            
        with open(covis_path, "rb") as f:
            covis_dict = pickle.load(f)

        sfm_model_path = scene_sfm / "sfm_superpoint+lightglue"
        sfm_cameras, sfm_images, sfm_points3D = rw.read_model(sfm_model_path, ext=".bin")

        pair_path = scene_covis / "most_similar_pairs.txt"
        pair_dict = load_similar_pairs(pair_path) if pair_path.exists() else {}

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
        
        # Trackers for MegaDepth
        mega_scene_metrics = {m: {t: [] for t in thresholds} for m in methods_to_plot}
        mega_scene_idx = []

        with h5py.File(q_feats_path, "r") as q_feats_h5, h5py.File(p3d_feats_path, "r") as p3d_feats_h5:
            
            desc_label = f"Computing Reproj Errors {scene}" if scene else "Computing Reproj Errors"
            for full_query_name in tqdm(covis_dict.keys(), desc=desc_label):
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

                camera_dict = None
                if args.dataset_type == "aachen":
                    if base_name not in gt_poses or full_query_name not in aachen_cams: continue
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
                    if full_query_name not in query_cams: continue
                    camera_dict = query_cams[full_query_name]

                visible_p3d = covis_dict[full_query_name]["unique_points"]
                if len(visible_p3d) == 0 or full_query_name not in q_feats_h5: continue

                q_kpts = q_feats_h5[full_query_name]["keypoints"][:]
                q_desc = q_feats_h5[full_query_name]["descriptors"][:]
                q_img_size = np.array(q_feats_h5[full_query_name]["image_size"][:])

                p3d_desc, p3d_xyz = [], []
                for pid in visible_p3d:
                    pid_str = str(pid)
                    if pid_str in p3d_feats_h5 and int(pid) in sfm_points3D:
                        p3d_desc.append(p3d_feats_h5[pid_str]["descriptors"][:].reshape(256))
                        p3d_xyz.append(sfm_points3D[int(pid)].xyz)
                        
                if len(p3d_xyz) == 0: continue
                p3d_desc = np.vstack(p3d_desc).T 
                p3d_xyz = np.vstack(p3d_xyz)   

                # Ground truth projection
                R = qvec2rotmat(camera_dict["qvec"])
                t = np.array(camera_dict["tvec"]).reshape(3, 1)
                params = camera_dict["intrinsics"]["params"]
                model_name = str(camera_dict["intrinsics"]["model"]).upper()
                
                if "SIMPLE" in model_name or "RADIAL" in model_name: 
                    fx, fy = params[0], params[0]
                    cx, cy = params[1], params[2]
                else:
                    fx, fy = params[0], params[1]
                    cx, cy = params[2], params[3]

                X = p3d_xyz.T 
                X_cam = R @ X + t 
                z_cam = X_cam[2]
                z_safe = np.where(z_cam <= 0, 1e-6, z_cam) 
                
                u = fx * (X_cam[0] / z_safe) + cx
                v = fy * (X_cam[1] / z_safe) + cy
                projected_true_2d = np.stack([u, v], axis=1)

                evaluated_queries += 1

                # Compute predictions
                ref_pose_matrix, ref_cam_obj = None, None
                ref_name = pair_dict.get(full_query_name)
                if ref_name:
                    ref_image_obj = next((img for img in sfm_images.values() if img.name == ref_name), None)
                    if ref_image_obj:
                        ref_R = qvec2rotmat(ref_image_obj.qvec)
                        ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))
                        ref_cam_obj = sfm_cameras[ref_image_obj.camera_id]

                preds = {}
                preds["NN"] = compute_nn_baseline(q_desc, p3d_desc, device)
                preds["TRAIN"] = compute_trained_lightglu3d(matchers['TRAIN'], q_kpts, q_desc, q_img_size, p3d_xyz, p3d_desc, device)
                preds["PR"], _, _, _, _ = compute_pr_baseline(matchers['BASELINE'], q_kpts, q_desc, q_img_size, p3d_xyz, p3d_desc, ref_pose_matrix, ref_cam_obj, device)


                if args.dataset_type == "aachen":
                    if category == "day": aachen_idx_day.append(q_index)
                    else: aachen_idx_night.append(q_index)
                else:
                    mega_scene_idx.append(q_index)

                # Calculate L2 distance and count survivors
                for m_name, pred_array in preds.items():
                    N_3D = len(p3d_xyz)
                    valid_mask = (pred_array > -1) & (pred_array < N_3D)
                    
                    dists = np.array([])
                    if np.any(valid_mask):
                        predicted_2d_kpts = q_kpts[valid_mask] 
                        matched_3d_indices = pred_array[valid_mask].astype(int) 
                        true_2d_kpts = projected_true_2d[matched_3d_indices]
                        valid_z_mask = z_cam[matched_3d_indices] > 0 
                        if np.any(valid_z_mask):
                            dists = np.linalg.norm(predicted_2d_kpts[valid_z_mask] - true_2d_kpts[valid_z_mask], axis=1)

                    # Count absolute points for each threshold
                    for t in thresholds:
                        survivor_count = np.sum(dists <= t) if len(dists) > 0 else 0
                        if args.dataset_type == "aachen":
                            if category == "day": aachen_day[m_name][t].append(survivor_count)
                            else: aachen_night[m_name][t].append(survivor_count)
                        else:
                            mega_scene_metrics[m_name][t].append(survivor_count)

        # Plotting
        colors = {"NN": "#e74c3c", "PR": "#f1c40f", "TRAIN": "#2ecc71"}
        X_STEP = 6 

        # Plot MegaDepth Scene
        if args.dataset_type == "megadepth" and mega_scene_idx:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax1 = plt.subplots(1, 1, figsize=(20, 7))
            sorted_idx = np.argsort(mega_scene_idx)
            real_q_idx = np.array(mega_scene_idx)[sorted_idx]
            
            for m in methods_to_plot:
                all_x, all_y = [], []
                for i in range(len(real_q_idx)):
                    base_x = i * X_STEP
                    all_x.extend([base_x, base_x + 1, base_x + 2, base_x + 3, np.nan])
                    all_y.extend([
                        mega_scene_metrics[m][3.0][sorted_idx[i]],
                        mega_scene_metrics[m][6.0][sorted_idx[i]],
                        mega_scene_metrics[m][9.0][sorted_idx[i]],
                        mega_scene_metrics[m][12.0][sorted_idx[i]],
                        np.nan
                    ])
                ax1.plot(all_x, all_y, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=3)
            
            ax1.axhline(y=15, color='r', linestyle='--', label='Danger Zone (<15 pts)')

            # Format X-Axis
            ax1.set_xticks(np.arange(len(real_q_idx)) * X_STEP + 1.5)
            ax1.set_xticklabels(real_q_idx, rotation=45, fontsize=9)
            
            ax1.set_title(f"MegaDepth {scene} - Cumulative Inliers Gradient [3, 6, 9, 12px]", fontsize=15, fontweight='bold')
            ax1.set_xlabel("Query Image Index", fontsize=13)
            ax1.set_ylabel("Absolute Match Count", fontsize=13)
            ax1.legend(fontsize=11)
            
            plt.tight_layout()
            plot_path = args.outputs / f"{args.dataset_type}_scene_{scene}_reprojection_sparklines.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved MegaDepth Scene plot to: {plot_path}")

    # Plot Aachen
    if args.dataset_type == "aachen":
        plt.style.use('seaborn-v0_8-darkgrid')
        X_STEP = 6
        
        # Day
        if aachen_idx_day:
            _, ax1 = plt.subplots(1, 1, figsize=(30, 7))
            sorted_idx = np.argsort(aachen_idx_day)
            real_q_idx = np.array(aachen_idx_day)[sorted_idx]
            
            for m in methods_to_plot:
                all_x, all_y = [], []
                for i in range(len(real_q_idx)):
                    base_x = i * X_STEP
                    all_x.extend([base_x, base_x + 1, base_x + 2, base_x + 3, np.nan])
                    all_y.extend([
                        aachen_day[m][3.0][sorted_idx[i]],
                        aachen_day[m][6.0][sorted_idx[i]],
                        aachen_day[m][9.0][sorted_idx[i]],
                        aachen_day[m][12.0][sorted_idx[i]],
                        np.nan
                    ])
                ax1.plot(all_x, all_y, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=2)
                
            ax1.axhline(y=15, color='r', linestyle='--', label='<15 pts')
            ax1.set_xticks(np.arange(len(real_q_idx)) * X_STEP + 1.5)
            
            # Hide some labels if there are over 800 queries to keep the bottom clean
            labels = [str(val) if i % 10 == 0 else "" for i, val in enumerate(real_q_idx)]
            ax1.set_xticklabels(labels, rotation=45, fontsize=8)
            ax1.set_title("Aachen Day - Cumulative Inliers Gradient [3, 6, 9, 12px]", fontsize=18, fontweight='bold')
            ax1.set_xlabel("Query Image Index", fontsize=14)
            ax1.set_ylabel("Absolute Match Count", fontsize=14)
            ax1.legend(fontsize=12)
            
            plt.tight_layout()
            plot_path_day = args.outputs / "aachen_day_reprojection_sparklines.png"
            plt.savefig(plot_path_day, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved Aachen Day plot to: {plot_path_day}")
        
        # Night
        if aachen_idx_night:
            _, ax2 = plt.subplots(1, 1, figsize=(16, 7)) 
            sorted_idx = np.argsort(aachen_idx_night)
            real_q_idx = np.array(aachen_idx_night)[sorted_idx]
            
            for m in methods_to_plot:
                all_x, all_y = [], []
                for i in range(len(real_q_idx)):
                    base_x = i * X_STEP
                    all_x.extend([base_x, base_x + 1, base_x + 2, base_x + 3, np.nan])
                    all_y.extend([
                        aachen_night[m][3.0][sorted_idx[i]],
                        aachen_night[m][6.0][sorted_idx[i]],
                        aachen_night[m][9.0][sorted_idx[i]],
                        aachen_night[m][12.0][sorted_idx[i]],
                        np.nan
                    ])
                ax2.plot(all_x, all_y, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=3)
                
            ax2.axhline(y=15, color='r', linestyle='--', label='<15 pts')
            ax2.set_xticks(np.arange(len(real_q_idx)) * X_STEP + 1.5)
            
            labels = [str(val) if i % 5 == 0 else "" for i, val in enumerate(real_q_idx)]
            ax2.set_xticklabels(labels, rotation=45, fontsize=9)
            ax2.set_title("Aachen Night - Cumulative Inliers Gradient [3, 6, 9, 12px]", fontsize=16, fontweight='bold')
            ax2.set_xlabel("Query Image Index", fontsize=13)
            ax2.set_ylabel("Absolute Match Count", fontsize=13)
            ax2.legend(fontsize=11)
            
            plt.tight_layout()
            plot_path_night = args.outputs / "aachen_night_reprojection_sparklines.png"
            plt.savefig(plot_path_night, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved Aachen Night plot to: {plot_path_night}")

if __name__ == "__main__":
    main()