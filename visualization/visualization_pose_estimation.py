# Visualize pose estimation.
# For baseline: NN(Nearest Neighbour), RR(Rotate+Remove_coord), 
#               RN(Rotate+Normalize), PR(Project to Reference),
#               HLOC(2D-2D LightGlue + Lift to 3D)  
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
import rerun as rr
from . import rerun_johanna as rru 
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat, get_most_similar_ref
from ground_truth.generate_gt_pairs_re import load_query_cams
from lightglue import LightGlue
import matplotlib.pyplot as plt
from baseline.pr_baseline import compute_pr_baseline
from baseline.rr_baseline import compute_rr_baseline
from baseline.rn_baseline import compute_rn_baseline
from baseline.nn_baseline import compute_nn_baseline
from baseline.trained_matcher import load_trained_lightglu3d, load_trained_adapt, compute_trained_lightglu3d
from evaluation.pose_estimation import compute_hloc_baseline, evaluate_pose
from baseline.pr_baseline_change import compute_pr_baseline_change

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockCamera:
    def __init__(self, width, height, params):
        self.size = [width, height]
        self.f = [params[0], params[1]]
        self.c = [params[2], params[3]]

def print_camera_pose(R_gt, t_gt, R_est, t_est):
    logger.info("="*40)
    logger.info("Camera Pose Comparison")
    logger.info("="*40)
    logger.info(f"Ground Truth tvec: {np.array2string(t_gt, precision=4)}")
    logger.info(f"Estimated tvec:    {np.array2string(t_est, precision=4)}")
    
    C_gt = -R_gt.T @ t_gt
    C_est = -R_est.T @ t_est
    t_err = np.linalg.norm(C_gt - C_est)

    delta_R = R_est @ R_gt.T
    trace = np.clip(np.trace(delta_R), -1.0, 3.0)
    r_err = np.degrees(np.arccos((trace - 1.0) / 2.0))

    logger.info("-" * 40)
    logger.info(f"Translation Error: {t_err:.4f} meters")
    logger.info(f"Rotation Error:    {r_err:.4f} degrees")
    logger.info("="*40)

def launch_rerun_visualization(pred_matches0, full_inlier_mask, q_kpts, p3d_kpts, raw_pts_np, raw_colors_np, scene, args, query_name, ref_name, camera, ref_pose_matrix, method_name, est_pose_matrix):
    query_stem = Path(query_name).stem
    logger.info(f"Initializing Rerun Analytics Dashboard for {method_name}...")
    rr.init(f"PoseVis_{scene}_{method_name}_{query_stem}", spawn=False)

    valid_pred = pred_matches0 > -1
    outlier_mask = valid_pred & (~full_inlier_mask)

    cameras, images, _ = rw.read_model(args.sfm_dir / scene / "sfm_superpoint+lightglue", ext=".bin")
    ref_image_obj = next((img for img in images.values() if img.name == ref_name), None)
    ref_cam_obj = cameras[ref_image_obj.camera_id]
    ref_poselib_cam = MockCamera(ref_cam_obj.width, ref_cam_obj.height, ref_cam_obj.params)

    q_pose_matrix = np.hstack((qvec2rotmat(camera["qvec"]), np.array(camera["tvec"]).reshape(3, 1)))
    query_poselib_cam = MockCamera(camera["intrinsics"]["width"], camera["intrinsics"]["height"], camera["intrinsics"]["params"])

    img_query = np.array(Image.open(args.dataset / scene / "images" / query_name).convert("RGB")) / 255.0
    img_ref = np.array(Image.open(args.dataset / scene / "images" / ref_name).convert("RGB")) / 255.0

    rru.plot_scene(
        pts_3d=np.empty((0,3)), pts_2d=np.empty((0,2)),           
        img_query=img_query, imgs_refs=[img_ref], 
        camera_poses_refs=np.array([ref_pose_matrix]), 
        poselib_cam_intrinsics_q=query_poselib_cam,
        poselib_cam_intrinsics_refs=[ref_poselib_cam], 
        cam_pose_query_estimated=est_pose_matrix,  
        cam_pose_query_gt=q_pose_matrix, 
        attach_image_to_est_pose=False  
    )

    # Log the 2D keypoints in the image plane
    img_path = "world/camera_query_gt/image"
    rr.log(f"{img_path}/Inliers", rr.Points2D(q_kpts[full_inlier_mask], colors=[0, 255, 0], radii=4.0))
    rr.log(f"{img_path}/Outliers", rr.Points2D(q_kpts[outlier_mask], colors=[255, 0, 0], radii=3.0)) 

    # Log the overall SfM Model
    rr.log("world/SfM_Context", rr.Points3D(raw_pts_np, colors=raw_colors_np, radii=0.03))
    
    # Calculate connections using the Estimated Camera Center
    if est_pose_matrix is not None:
        est_cam_center = (-est_pose_matrix[:, :3].T @ est_pose_matrix[:, 3]).flatten()

        # Inliers (Green)
        inlier_3d_pts = p3d_kpts[pred_matches0[full_inlier_mask]]
        inlier_lines = [[est_cam_center, pt] for pt in inlier_3d_pts]
        rr.log("world/PnP/Inliers/Points", rr.Points3D(inlier_3d_pts, colors=[0, 255, 0], radii=0.06))
        rr.log("world/PnP/Inliers/Lines", rr.LineStrips3D(inlier_lines, colors=[0, 255, 0, 100]))
        
        # Outliers (Red)
        outlier_3d_pts = p3d_kpts[pred_matches0[outlier_mask]]
        outlier_lines = [[est_cam_center, pt] for pt in outlier_3d_pts]
        rr.log("world/PnP/Outliers/Points", rr.Points3D(outlier_3d_pts, colors=[255, 0, 0], radii=0.06))
        rr.log("world/PnP/Outliers/Lines", rr.LineStrips3D(outlier_lines, colors=[255, 0, 0, 50]))

    output_filename = f"viz_{scene}_{method_name}_{query_stem}.rrd"
    rr.save(output_filename)
    logger.info(f"Rerun visualization saved to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Matches & Pose in Rerun")
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--covisibility_dir', type=Path, required=True)
    parser.add_argument('--query_dir', type=Path, required=True)
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--method', type=str, required=True, choices=['NN', 'RR', 'RN', 'PR', 'PRC', 'TRAIN', 'ADAPT'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--max_error', type=float, default=3.0, help="RANSAC Reprojection Error Threshold (pixels)")
    parser.add_argument('--num_hloc_refs', type=int, default=5, help="Number of reference images for HLOC aggregation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = args.scene
    method = args.method
    logger.info(f"Starting Evaluation & Visualization for Scene {scene} using method: {method}")

    # Random query selection
    query_names_file = args.query_dir / scene / "query_image_names_clean.txt"
    with open(query_names_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    query_name = random.choice(queries)
    
    active_pair_file = args.covisibility_dir / scene / "most_similar_pairs.txt"

    # Prepare reference for HLOC
    top_refs = []
    if method == "HLOC" and args.num_hloc_refs > 1:
        logger.info(f"Loading NetVLAD features into memory for Top-{args.num_hloc_refs} retrieval...")
        queries_features = args.covisibility_dir / scene / 'feats-netvlad-query.h5'
        references_features = args.covisibility_dir / scene / 'feats-netvlad-ref.h5'
        
        if queries_features.exists() and references_features.exists():
            with h5py.File(references_features, 'r') as f_ref:
                ref_names_list = [n for n in f_ref.keys() if 'global_descriptor' in f_ref[n]]
                if ref_names_list:
                    descs = [torch.from_numpy(f_ref[n]['global_descriptor'][:]).view(-1) for n in ref_names_list]
                    ref_descs_tensor = torch.stack(descs).to(device)
                    
                    with h5py.File(queries_features, 'r') as f_q:
                        if query_name in f_q and 'global_descriptor' in f_q[query_name]:
                            q_desc_vlad = torch.from_numpy(f_q[query_name]['global_descriptor'][:]).view(-1).to(device)
                            sim = ref_descs_tensor @ q_desc_vlad 
                            k = min(args.num_hloc_refs, len(ref_names_list))
                            topk_idx = torch.topk(sim, k).indices.cpu().numpy()
                            top_refs = [ref_names_list[i] for i in topk_idx]
        else:
            logger.warning("NetVLAD feature files missing. Falling back to most_similar_pairs.txt")
            
    if not top_refs:
        ref_name = get_most_similar_ref(query_name, active_pair_file)
        top_refs = [ref_name] if ref_name else []

    if not top_refs:
        logger.error("No valid references found.")
        return

    primary_ref = top_refs[0]
    logger.info(f"Query: {query_name} | Primary Reference: {primary_ref}")

    # Load query image and features
    img_query_pil = Image.open(args.dataset / scene / "images" / query_name)
    q_img_size = [img_query_pil.width, img_query_pil.height]
    
    features_path = args.sfm_dir / scene / "feats-superpoint-n2048.h5"
    with h5py.File(features_path, "r") as f:
        q_kpts = f[query_name]["keypoints"][:]
        q_desc = f[query_name]["descriptors"][:]

    # Load 3D points
    sfm_model_path = args.sfm_dir / scene / "sfm_superpoint+lightglue"
    reconstruction = pycolmap.Reconstruction(sfm_model_path)
    
    cameras, images, _ = rw.read_model(sfm_model_path, ext=".bin")
    ref_image_obj = next((img for img in images.values() if img.name == primary_ref), None)
    ref_R = qvec2rotmat(ref_image_obj.qvec)
    ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))
    ref_cam_obj = cameras[ref_image_obj.camera_id]

    with open(args.covisibility_dir / scene / "covisibility_results.pkl", "rb") as f:
        visible_p3d = pickle.load(f)[query_name]["unique_points"]

    p3d_desc, p3d_kpts, raw_colors = [], [], []
    p3d_indices_map = {} 

    with h5py.File(args.covisibility_dir / scene / "points3D_feats_cache.h5", "r") as f:
        idx_counter = 0
        for pid in visible_p3d:
            pid_int = int(pid)
            if str(pid) in f and pid_int in reconstruction.points3D:
                p3d_desc.append(f[str(pid)]["descriptors"][:].reshape(256))
                p3d_kpts.append(f[str(pid)]["keypoints"][:].reshape(3))
                raw_colors.append(reconstruction.points3D[pid_int].color)
                p3d_indices_map[pid_int] = idx_counter
                idx_counter += 1
                
    if not p3d_kpts:
        logger.error("No valid 3D coordinates/features found.")
        return

    p3d_desc = np.vstack(p3d_desc).T 
    p3d_kpts = np.vstack(p3d_kpts)   
    raw_pts_np = p3d_kpts.copy() 
    raw_colors_np = np.vstack(raw_colors) / 255.0

    query_cams = load_query_cams(args.query_dir / scene / "query_image_cameras.txt")
    camera = query_cams[query_name]

    # Initialize Models
    if method in ["RR", "RN", "PR", "PRC", "HLOC"]:
        baseline_matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    if method == "TRAIN":
        lightglu3d_matcher = load_trained_lightglu3d(args.checkpoint, device)
    if method == "ADAPT":
        lightglu3d_adapt_matcher = load_trained_adapt(args.checkpoint, device)

    # Prepare reference data for HLOC
    ref_data_list = []
    if method == "HLOC":
        with h5py.File(features_path, "r") as f:
            for r_name in top_refs:
                if r_name in f:
                    r_img_pil = Image.open(args.dataset / scene / "images" / r_name)
                    r_img_obj = next((img for img in images.values() if img.name == r_name), None)
                    if r_img_obj is not None:
                        ref_data_list.append({
                            "kpts": f[r_name]["keypoints"][:], "desc": f[r_name]["descriptors"][:],
                            "img_size": [r_img_pil.width, r_img_pil.height], "p3d_ids": r_img_obj.point3D_ids,
                            "name": r_name, "image_id": r_img_obj.id
                        })

    # Get Predictions
    if method == "HLOC":
        # FIXED BUG: Passed p3d_indices_map instead of reconstruction
        pred_matches0 = compute_hloc_baseline(baseline_matcher, q_kpts, q_desc, q_img_size, ref_data_list, p3d_indices_map, device)
    else:
        if method == "NN":
            pred_matches0 = compute_nn_baseline(q_desc, p3d_desc, device)
        elif method == "RR":
            pred_matches0, _, _, _, _ = compute_rr_baseline(baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device)
        elif method == "RN":
            pred_matches0, _, _, _, _ = compute_rn_baseline(baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device)
        elif method == "PR":
            pred_matches0, _, _, _, _ = compute_pr_baseline(baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, ref_cam_obj, device)
        elif method == "PRC":
            pred_matches0, _, _, _, _ = compute_pr_baseline_change(baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, camera, device)
        elif method == "TRAIN":
            pred_matches0, _ = compute_trained_lightglu3d(lightglu3d_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)
        elif method == "ADAPT":
            pred_matches0, _ = compute_trained_lightglu3d(lightglu3d_adapt_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)

    # Evaluate Pose
    valid_mask = pred_matches0 > -1
    full_inlier_mask = np.zeros(len(q_kpts), dtype=bool)
    est_pose_matrix = None
    
    matched_2d = q_kpts[valid_mask]
    matched_3d = p3d_kpts[pred_matches0[valid_mask]]
    
    pose_res = evaluate_pose(matched_2d, matched_3d, camera, args.max_error)

    if pose_res is not None:
        est_pose_matrix = np.hstack((pose_res["R_est"], pose_res["t_est"].reshape(3, 1)))
        
        # Maps the truncated RANSAC inlier mask back to the original 2048-length keypoint array
        full_inlier_mask[valid_mask] = pose_res["inliers"]
        
        # Log Summary Counts
        total_guesses = valid_mask.sum()
        total_inliers = pose_res["inliers"].sum()
        total_outliers = total_guesses - total_inliers
        logger.info(f"RANSAC Results: {total_inliers} Inliers / {total_guesses} Guessed Matches ({total_outliers} Outliers Rejected)")

        R_gt = qvec2rotmat(camera["qvec"])
        t_gt = np.array(camera["tvec"]).reshape(3)
        print_camera_pose(R_gt, t_gt, pose_res["R_est"], pose_res["t_est"])
    else:
        logger.warning("PnP failed to find a valid pose (too many outliers or insufficient matches).")

    launch_rerun_visualization(
        pred_matches0=pred_matches0, 
        full_inlier_mask=full_inlier_mask,  
        q_kpts=q_kpts, p3d_kpts=p3d_kpts, 
        raw_pts_np=raw_pts_np, raw_colors_np=raw_colors_np,
        scene=scene, args=args, query_name=query_name, ref_name=primary_ref, 
        camera=camera,
        ref_pose_matrix=ref_pose_matrix,
        method_name=method,
        est_pose_matrix=est_pose_matrix
    )

if __name__ == "__main__":
    main()