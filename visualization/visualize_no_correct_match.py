# For baseline: NN(Nearest Neighbour), RR(Rotate+Remove_coord), 
#               RN(Rotate+Normalize), PR(Project to Reference)
# For train: TRAIN(Lightglu3d two self and one bidirectional cross), 
#            ADAPT(lightglue+adapter)

# Before use it, add the gluefactory path in the terminal
# export PYTHONPATH="/home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match:$PYTHONPATH"

import argparse
import logging
import pickle
import random
import numpy as np
import torch
import h5py
from pathlib import Path
from PIL import Image
import pycolmap
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat, get_most_similar_ref
from ground_truth.generate_gt_pairs_re import load_query_cams, compute_ground_truth_matches
from lightglue import LightGlue
from baseline.pr_baseline import compute_pr_baseline
from baseline.rr_baseline import compute_rr_baseline, compute_precision_recall
from baseline.rn_baseline import compute_rn_baseline
from baseline.nn_baseline import compute_nn_baseline
from baseline.trained_matcher import load_trained_lightglu3d, load_trained_adapt, compute_trained_lightglu3d
from evaluation.pose_estimation_aachen import parse_aachen_cameras, load_reference_poses
from ground_truth.generate_ref_gt_pairs_from_hloc_aachen import compute_ground_truth_matches_aachen
from .visualize_matches import launch_rerun_visualization, get_image_path, visual_flat_sfm

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Visualize Matches in Rerun")
    parser.add_argument('--dataset_type', type=str, default='megadepth', choices=['aachen', 'megadepth'])
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Undistorted_SfM or Aachen dataset root")
    parser.add_argument('--image_dir', type=Path, default=None, help="Explicit path to images directory (useful for Aachen unzipped images)")
    parser.add_argument('--covisibility_dir', type=Path, required=True, help="Path to covisibility")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query directory")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to sfm outputs")
    parser.add_argument('--depth_dir', type=Path, help="Path to depth maps (Required for MegaDepth)")
    parser.add_argument('--hloc_reference', type=Path, help="Path to hloc output (Required for Aachen GT)")
    parser.add_argument('--scene', type=str, default="", help="Scene name (or 'day'/'night' for Aachen)")
    parser.add_argument('--query_name', type=str, default=None, help="Optional: Specific query image name to visualize")
    parser.add_argument('--method', type=str, required=True, choices=['NN', 'RR', 'RN', 'PR', 'TRAIN', 'ADAPT'], 
                        help="Matching method to evaluate: NN, RR, RN, PR, TRAIN or ADAPT")
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help="Path to trained network weights (Only required if method is TRAIN or ADAPT)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = args.scene
    method = args.method
    scene_label = scene if scene else "Aachen"
    logger.info(f"Starting Evaluation & Visualization for {scene_label} using method: {method}")

    scene_query = args.query_dir / scene if (scene and args.query_dir and args.dataset_type != "aachen") else args.query_dir
    scene_covis = args.covisibility_dir / scene if (scene and args.dataset_type != "aachen") else args.covisibility_dir
    scene_sfm = args.sfm_dir / scene if (scene and args.dataset_type != "aachen") else args.sfm_dir

    # Load Models
    if method in ["RR", "RN", "PR"]:
        logger.info("Initializing standard LightGlue for baseline evaluation...")
        baseline_matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    elif method == "TRAIN":
        if args.checkpoint is None: raise ValueError("--checkpoint must be provided when using the TRAIN method.")
        lightglu3d_matcher = load_trained_lightglu3d(args.checkpoint, device)
    elif method == "ADAPT":
        if args.checkpoint is None: raise ValueError("--checkpoint must be provided when using the ADAPT method.")
        lightglu3_adapt_matcher = load_trained_adapt(args.checkpoint, device)

    # Load SfM Model
    sfm_model_path = scene_sfm / "sfm_superpoint+lightglue"
    reconstruction = pycolmap.Reconstruction(sfm_model_path)
    cameras, images, _ = rw.read_model(sfm_model_path, ext=".bin")

    # Load Covisibility
    with open(scene_covis / "covisibility_results.pkl", "rb") as f:
        covis_dict = pickle.load(f)

    # Load Queries List
    queries = []
    if args.dataset_type == "aachen":
        day_list_path = scene_query / "day_time_queries_with_intrinsics.txt"
        night_list_path = scene_query / "night_time_queries_with_intrinsics.txt"
        if args.scene in ["day", ""] and day_list_path.exists():
            with open(day_list_path, 'r') as f:
                queries.extend([line.strip().split()[0] for line in f if line.strip() and not line.startswith("#")])
        if args.scene in ["night", ""] and night_list_path.exists():
            with open(night_list_path, 'r') as f:
                queries.extend([line.strip().split()[0] for line in f if line.strip() and not line.startswith("#")])
    else:
        query_names_file = scene_query / "query_image_names_clean.txt"
        with open(query_names_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]

    # Parse GT Cams for MegaDepth/Aachen Day-Night once
    if args.dataset_type == "aachen":
        aachen_cams = parse_aachen_cameras(scene_query)
        gt_poses = load_reference_poses(args.hloc_reference)
    else:
        query_cams = load_query_cams(scene_query / "query_image_cameras.txt")

    # Prepare search list
    if args.query_name:
        matched_queries = [q for q in queries if args.query_name in q]
        if not matched_queries:
            logger.error(f"Provided query '{args.query_name}' not found.")
            return
        queries_to_eval = [matched_queries[0]] # Just evaluate the one requested
    else:
        queries_to_eval = queries.copy()
        random.shuffle(queries_to_eval) # Shuffle so we find a random broken query

    # Iterate to find target query (num_correct == 0)

    found_target = False
    
    for query_name in queries_to_eval:
        ref_name = get_most_similar_ref(query_name, scene_covis / "most_similar_pairs.txt")
        if not ref_name: continue

        # Resolve Image Paths
        img_query_path = get_image_path(args.dataset, args.dataset_type, scene, query_name, args.image_dir)
        img_ref_path = get_image_path(args.dataset, args.dataset_type, scene, ref_name, args.image_dir)

        # Load query image
        img_query_pil = Image.open(img_query_path)
        q_img_size = np.array([img_query_pil.width, img_query_pil.height])

        # Load features
        with h5py.File(scene_sfm / "feats-superpoint-n2048.h5", "r") as f:
            if query_name not in f: continue
            q_kpts = f[query_name]["keypoints"][:]
            q_desc = f[query_name]["descriptors"][:]

        # Load visible 3d points
        visible_p3d = covis_dict.get(query_name, {}).get("unique_points", [])
        if len(visible_p3d) == 0: continue

        p3d_desc, p3d_kpts, raw_colors = [], [], []
        with h5py.File(scene_covis / "points3D_feats_cache.h5", "r") as f:
            for pid in visible_p3d:
                pid_int = int(pid)
                pid_str = str(pid)
                if pid_str in f and pid_int in reconstruction.points3D:
                    p3d_desc.append(f[pid_str]["descriptors"][:].reshape(256))
                    p3d_kpts.append(f[pid_str]["keypoints"][:].reshape(3))
                    raw_colors.append(reconstruction.points3D[pid_int].color)
                    
        if not p3d_kpts: continue

        p3d_desc = np.vstack(p3d_desc).T 
        p3d_kpts = np.vstack(p3d_kpts)   
        raw_pts_np = p3d_kpts.copy() 
        raw_colors_np = np.vstack(raw_colors) / 255.0

        # Calculate ground truth dynamically
        if args.dataset_type == "aachen":
            base_name = Path(query_name).name
            if query_name not in aachen_cams or base_name not in gt_poses: continue
            raw_camera = aachen_cams[query_name]
            gt_pose = gt_poses[base_name]
            camera = {
                "qvec": gt_pose["qvec"], "tvec": gt_pose["tvec"],
                "intrinsics": {
                    "model": getattr(raw_camera, 'model_name', getattr(raw_camera.model, 'name', str(raw_camera.model))),
                    "width": raw_camera.width, "height": raw_camera.height, "params": raw_camera.params
                }
            }
            gt_matches0, _ = compute_ground_truth_matches_aachen(
                q_kpts, p3d_kpts, raw_camera, gt_pose, pos_reproj_thresh=3.0, neg_reproj_thresh=5.0
            )
        else:
            if query_name not in query_cams: continue
            camera = query_cams[query_name]
            with h5py.File(args.depth_dir / scene / f"{Path(query_name).stem}.h5", 'r') as f:
                depth_map = f['depth'][:]
            gt_matches0, _ = compute_ground_truth_matches(
                {"keypoints": q_kpts}, {"keypoints": p3d_kpts}, camera, depth_map
            )

        # Load Reference Camera Pose
        ref_image_obj = next((img for img in images.values() if img.name == ref_name), None)
        if not ref_image_obj: continue
        ref_R = qvec2rotmat(ref_image_obj.qvec)
        ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))
        ref_cam_obj = cameras[ref_image_obj.camera_id]

        # Get Predicted Matches
        if method == "NN":
            pred_matches0 = compute_nn_baseline(q_desc, p3d_desc, device)
        elif method == "RR":
            pred_matches0, res, p3d_flat_kpts, flat_w, flat_h = compute_rr_baseline(
                baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device
            )
        elif method == "RN":
            pred_matches0, res, p3d_flat_kpts, flat_w, flat_h = compute_rn_baseline(
                baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device
            )
        elif method == "PR":
            pred_matches0, res, p3d_flat_kpts, flat_w, flat_h = compute_pr_baseline(
                baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, ref_cam_obj, device
            )
        elif method == "TRAIN":
            pred_matches0 = compute_trained_lightglu3d(lightglu3d_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)
        elif method == "ADAPT":
            pred_matches0 = compute_trained_lightglu3d(lightglu3_adapt_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)
        
        # Evaluate metrics
        precision, recall, num_gt, num_pred, num_correct = compute_precision_recall(pred_matches0, gt_matches0)

        # BREAK CONDITION: If user specified query, OR we found 0 correct matches
        if args.query_name or num_correct == 0:
            found_target = True
            logger.info(f"Target found -> Query: {query_name} | Reference: {ref_name}")
            break
        else:
            logger.info(f"Scanning... Skipped {query_name} (It had {num_correct} correct matches)")

    if not found_target:
        logger.info(f"All evaluated queries in '{scene_label}' have at least 1 correct match. No completely failed query found.")
        return

    disp_precision = precision if precision is not None else 0.0
    disp_recall = recall if recall is not None else 0.0

    logger.info("="*30)
    logger.info(f"{method} Results for {query_name}:")
    logger.info(f"GT Matches:        {num_gt}")
    logger.info(f"Predicted Matches: {num_pred}")
    logger.info(f"Correct Matches:   {num_correct}")
    logger.info(f"Precision:         {disp_precision:.4f}")
    logger.info(f"Recall:            {disp_recall:.4f}")
    logger.info("="*30)

    # Launch 2D flat images
    if method in ["RR", "RN", "PR"]:
        img_query_np = np.array(img_query_pil.convert("RGB")) / 255.0
        visual_flat_sfm(res, q_kpts, p3d_flat_kpts, img_query_np, raw_colors_np, flat_w, flat_h, scene, method)

    # Launch rerun
    launch_rerun_visualization(
        pred_matches0=pred_matches0, 
        gt_matches0=gt_matches0,  
        q_kpts=q_kpts, p3d_kpts=p3d_kpts, 
        raw_pts_np=raw_pts_np, raw_colors_np=raw_colors_np,
        scene=scene, args=args, query_name=query_name, ref_name=ref_name, 
        camera=camera,
        ref_pose_matrix=ref_pose_matrix,
        img_query_path=img_query_path,
        img_ref_path=img_ref_path,
        dataset_type=args.dataset_type,
        method_name=method
    )

if __name__ == "__main__":
    main()