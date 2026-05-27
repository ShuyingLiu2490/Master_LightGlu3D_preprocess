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
from baseline.rr_baseline import load_similar_pairs, compute_precision_recall
from baseline.pr_baseline import compute_pr_baseline
from ground_truth.generate_ref_gt_pairs_from_hloc_aachen import compute_ground_truth_matches_aachen
from ground_truth.generate_gt_pairs_by_scene import compute_ground_truth_matches_soft, load_query_cams
from evaluation.pose_estimation_aachen import load_reference_poses, parse_aachen_cameras

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate Precision & Recall Statistics by Query Index")
    parser.add_argument('--dataset_type', type=str, default='aachen', choices=['aachen', 'megadepth'])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--covisibility_dir', type=Path, required=True)
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True, help="Directory to save the plotted figure")
    parser.add_argument('--checkpoint_train', type=str, required=True, help="Path to LightGlu3D weights")
    parser.add_argument('--checkpoint_adapt', type=str, default=None, help="Path to LightGlue_Adapt weights (unused)")
    parser.add_argument('--hloc_reference', type=Path, help="Required for Aachen GT poses")
    parser.add_argument('--query_dir', type=Path, help="Required for MegaDepth custom query locations")
    parser.add_argument('--scene_list', type=Path, help="Required for MegaDepth scene loops")
    parser.add_argument('--depth_dir', type=Path, help="Required for MegaDepth depth maps")
    args = parser.parse_args()

    stat_dir = args.outputs
    stat_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Initializing NN, PR, and TRAIN Matchers...")
    
    matchers = {"NN": None, 
                "TRAIN": load_trained_lightglu3d(args.checkpoint_train, device),
                "BASELINE": LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)}
    
    # Three matchers
    methods_to_plot = ["NN", "PR", "TRAIN"]
    colors = {"NN": "#e74c3c", "PR": "#f1c40f", "TRAIN": "#2ecc71"}

    logger.info(f"Loading {args.dataset_type.upper()} dataset structures...")
    if args.dataset_type == "megadepth":
        if not args.query_dir or not args.scene_list or not args.depth_dir:
            raise ValueError("--query_dir, --scene_list, and --depth_dir are strictly required for MegaDepth.")
        with open(args.scene_list, 'r') as f:
            scenes = [line.strip() for line in f if line.strip()]
    else:
        scenes = [""] 

    # Trackers for Aachen
    aachen_metrics_day = {m: {'precision': [], 'recall': []} for m in methods_to_plot}
    aachen_metrics_night = {m: {'precision': [], 'recall': []} for m in methods_to_plot}
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

        # Load exact query lists to determine order/index
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
        mega_scene_metrics = {m: {'precision': [], 'recall': []} for m in methods_to_plot}
        mega_scene_idx = []

        with h5py.File(q_feats_path, "r") as q_feats_h5, h5py.File(p3d_feats_path, "r") as p3d_feats_h5:
            
            desc_label = f"Evaluating Index {scene}" if scene else "Evaluating Index"
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

                camera_dict, raw_camera, gt_pose = None, None, None
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
                    raw_camera = camera_dict
                    gt_pose = None

                visible_p3d = covis_dict[full_query_name]["unique_points"]
                if len(visible_p3d) == 0 or full_query_name not in q_feats_h5: continue

                q_kpts = q_feats_h5[full_query_name]["keypoints"][:]
                q_desc = q_feats_h5[full_query_name]["descriptors"][:]
                q_img_size = torch.from_numpy(np.array(q_feats_h5[full_query_name]["image_size"][:])).float()

                p3d_desc, p3d_xyz = [], []
                for pid in visible_p3d:
                    pid_str = str(pid)
                    if pid_str in p3d_feats_h5 and int(pid) in sfm_points3D:
                        p3d_desc.append(p3d_feats_h5[pid_str]["descriptors"][:].reshape(256))
                        p3d_xyz.append(sfm_points3D[int(pid)].xyz)
                        
                if len(p3d_xyz) == 0: continue
                p3d_desc = np.vstack(p3d_desc).T 
                p3d_xyz = np.vstack(p3d_xyz)   

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

                # Compute ground truth
                if args.dataset_type == "aachen":
                    gt0, _ = compute_ground_truth_matches_aachen(q_kpts, p3d_xyz, raw_camera, gt_pose, pos_reproj_thresh=3.0, neg_reproj_thresh=5.0)
                else:
                    gt0, _ = compute_ground_truth_matches_soft({"keypoints": q_kpts}, {"keypoints": p3d_xyz}, camera_dict, depth_map=depth_map, 
                        pos_reproj_thresh=3.0, neg_reproj_thresh=5.0, pos_depth_thresh=0.1, neg_depth_thresh=0.25)
                if np.sum(gt0 >= 0) == 0:
                    continue

                # Compute predict matches
                preds = {}
                preds["NN"] = compute_nn_baseline(q_desc, p3d_desc, device)
                preds["TRAIN"], _ = compute_trained_lightglu3d(matchers['TRAIN'], q_kpts, q_desc, q_img_size.numpy(), p3d_xyz, p3d_desc, device)
                preds["PR"], _, _, _, _ = compute_pr_baseline(matchers['BASELINE'], q_kpts, q_desc, q_img_size.numpy(), p3d_xyz, p3d_desc, ref_pose_matrix, ref_cam_obj, device)

                if args.dataset_type == "aachen":
                    if category == "day": aachen_idx_day.append(q_index)
                    else: aachen_idx_night.append(q_index)
                else:
                    mega_scene_idx.append(q_index)

                for m_name, pred_array in preds.items():
                    precision, recall, _, _, _= compute_precision_recall(pred_array, gt0)
                    
                    if precision is None: precision = 0.0
                    if recall is None: recall = 0.0
                    
                    if args.dataset_type == "aachen":
                        if category == "day": 
                            aachen_metrics_day[m_name]['precision'].append(precision)
                            aachen_metrics_day[m_name]['recall'].append(recall)
                        else: 
                            aachen_metrics_night[m_name]['precision'].append(precision)
                            aachen_metrics_night[m_name]['recall'].append(recall)
                    else:
                        mega_scene_metrics[m_name]['precision'].append(precision)
                        mega_scene_metrics[m_name]['recall'].append(recall)

        # Plot MegaDepth 
        if args.dataset_type == "megadepth" and mega_scene_idx:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
            sorted_idx = np.argsort(mega_scene_idx)
            x_vals = np.array(mega_scene_idx)[sorted_idx]
            
            for m in methods_to_plot:
                y_prec = np.array(mega_scene_metrics[m]['precision'])[sorted_idx]
                y_rec = np.array(mega_scene_metrics[m]['recall'])[sorted_idx]
                ax1.plot(x_vals, y_prec, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=4)
                ax2.plot(x_vals, y_rec, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=4)
            
            ax1.set_title(f"MegaDepth Scene {scene} - Precision", fontsize=15, fontweight='bold')
            ax1.set_xlabel("Query Image Index", fontsize=13)
            ax1.set_ylabel("Match Precision", fontsize=13)
            ax1.legend(fontsize=11)
            
            ax2.set_title(f"MegaDepth Scene {scene} - Recall", fontsize=15, fontweight='bold')
            ax2.set_xlabel("Query Image Index", fontsize=13)
            ax2.set_ylabel("Match Recall", fontsize=13)
            ax2.legend(fontsize=11)
            
            plt.tight_layout()
            plot_path = stat_dir / f"megadepth_scene_{scene}_index_statistics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved MegaDepth Scene plot to: {plot_path}")

    # Plot Aachen
    if args.dataset_type == "aachen":
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12), sharey='col')
        
        # Day queries
        if aachen_idx_day:
            sorted_idx = np.argsort(aachen_idx_day)
            x_vals = np.array(aachen_idx_day)[sorted_idx]
            for m in methods_to_plot:
                y_prec = np.array(aachen_metrics_day[m]['precision'])[sorted_idx]
                y_rec = np.array(aachen_metrics_day[m]['recall'])[sorted_idx]
                ax1.plot(x_vals, y_prec, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=4)
                ax2.plot(x_vals, y_rec, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=4)
                
        ax1.set_title("Aachen Day Queries - Precision", fontsize=15, fontweight='bold')
        ax1.set_ylabel("Match Precision", fontsize=13)
        ax1.legend(fontsize=11)
        
        ax2.set_title("Aachen Day Queries - Recall", fontsize=15, fontweight='bold')
        ax2.set_ylabel("Match Recall", fontsize=13)
        ax2.legend(fontsize=11)
        
        # Night queries
        if aachen_idx_night:
            sorted_idx = np.argsort(aachen_idx_night)
            x_vals = np.array(aachen_idx_night)[sorted_idx]
            for m in methods_to_plot:
                y_prec = np.array(aachen_metrics_night[m]['precision'])[sorted_idx]
                y_rec = np.array(aachen_metrics_night[m]['recall'])[sorted_idx]
                ax3.plot(x_vals, y_prec, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=4)
                ax4.plot(x_vals, y_rec, label=m, color=colors[m], alpha=0.8, linewidth=1.5, marker='o', markersize=4)
                
        ax3.set_title("Aachen Night Queries - Precision", fontsize=15, fontweight='bold')
        ax3.set_xlabel("Query Image Index", fontsize=13)
        ax3.set_ylabel("Match Precision", fontsize=13)
        ax3.legend(fontsize=11)

        ax4.set_title("Aachen Night Queries - Recall", fontsize=15, fontweight='bold')
        ax4.set_xlabel("Query Image Index", fontsize=13)
        ax4.set_ylabel("Match Recall", fontsize=13)
        ax4.legend(fontsize=11)
        
        plt.tight_layout()
        plot_path = stat_dir / "aachen_index_statistics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Aachen plot to: {plot_path}")

if __name__ == "__main__":
    main()