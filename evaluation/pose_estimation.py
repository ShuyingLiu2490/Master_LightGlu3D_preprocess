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
import numpy as np
import torch
import h5py
from pathlib import Path
from PIL import Image
import pycolmap
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat, get_most_similar_ref
from tqdm import tqdm
from lightglue import LightGlue
from baseline.pr_baseline import compute_pr_baseline
from baseline.rr_baseline import compute_rr_baseline
from baseline.rn_baseline import compute_rn_baseline
from ground_truth.generate_gt_pairs_soft import load_query_cams
from baseline.nn_baseline import compute_nn_baseline
from baseline.trained_matcher import load_trained_lightglu3d, load_trained_adapt, compute_trained_lightglu3d

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_pose(matched_2d, matched_3d, camera, q_img_size, max_error): 
    if len(matched_2d) < 4:
        return None

    orig_w = camera["intrinsics"]["width"]
    orig_h = camera["intrinsics"]["height"]
    new_w, new_h = q_img_size[0], q_img_size[1]
    
    params = np.array(camera["intrinsics"]["params"], dtype=float)
    
    if orig_w != new_w or orig_h != new_h:
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        model = camera["intrinsics"]["model"]
        
        if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
            params[0] *= scale_x # f
            params[1] *= scale_x # cx
            params[2] *= scale_y # cy
        elif model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"]:
            params[0] *= scale_x # fx
            params[1] *= scale_y # fy
            params[2] *= scale_x # cx
            params[3] *= scale_y # cy

    colmap_cam = pycolmap.Camera(
        model=camera["intrinsics"]["model"],
        width=int(new_w),
        height=int(new_h),
        params=params
    )

    estimation_options = {"ransac": {"max_error": max_error}}
    refinement_options = {"refine_focal_length": False, "refine_extra_params": False}

    # Run Absolute Pose Estimation (PnP + RANSAC)
    ret = pycolmap.estimate_and_refine_absolute_pose(
        matched_2d, matched_3d, colmap_cam, estimation_options, refinement_options
    )

    if ret is None or not ret.get("is_valid", True):
        return None

    # Handle different pycolmap version returns safely
    if isinstance(ret, dict):
        R_est = ret["cam_from_world"].rotation.matrix()
        t_est = ret["cam_from_world"].translation
        inliers = ret["inlier_mask"]
    else:
        R_est = qvec2rotmat(ret.qvec)
        t_est = ret.tvec
        inliers = ret.inlier_mask

    # Extract Ground Truth Pose
    R_gt = qvec2rotmat(camera["qvec"])
    t_gt = np.array(camera["tvec"]).reshape(3)

    # Calculate Translation Error (Distance between camera centers)
    C_gt = -R_gt.T @ t_gt
    C_est = -R_est.T @ t_est
    t_error = np.linalg.norm(C_est - C_gt)
    
    # Calculate Rotation Error (Angle difference in degrees)
    delta_R = R_est @ R_gt.T
    trace = np.clip(np.trace(delta_R), -1.0, 3.0)
    r_error_deg = np.degrees(np.arccos((trace - 1.0) / 2.0))

    return {
        "t_error": t_error,
        "r_error_deg": r_error_deg,
        "R_est": R_est,
        "t_est": t_est,
        "inliers": inliers
    }

def compute_hloc_baseline(baseline_matcher, q_kpts, q_desc, q_img_size, ref_data_list, p3d_indices_map, device):
    pred_matches0 = np.full(len(q_kpts), -1)

    for ref in ref_data_list:
        data = {
            "image0": {
                "keypoints": torch.from_numpy(q_kpts).unsqueeze(0).float().to(device),
                "descriptors": torch.from_numpy(q_desc.T).unsqueeze(0).float().to(device), 
                "image_size": torch.tensor([q_img_size]).float().to(device)
            },
            "image1": {
                "keypoints": torch.from_numpy(ref["kpts"]).unsqueeze(0).float().to(device),
                "descriptors": torch.from_numpy(ref["desc"].T).unsqueeze(0).float().to(device),
                "image_size": torch.tensor([ref["img_size"]]).float().to(device)
            }
        }
        with torch.no_grad():
            res = baseline_matcher(data)
        
        matches_2d = res['matches'][0].cpu().numpy() 
        ref_points3D_ids = ref["p3d_ids"]

        for q_idx, r_idx in matches_2d:
            if pred_matches0[q_idx] == -1:
                p3d_id = int(ref_points3D_ids[r_idx])
                
                # Handle invalid points
                if p3d_id != -1 and p3d_id != 18446744073709551615:
                    if p3d_id in p3d_indices_map:
                        pred_matches0[q_idx] = p3d_indices_map[p3d_id]

    return pred_matches0

def process_scene(scene, args, matchers, device, is_megadepth=True):
    t_errors = []
    r_errors = []
    failed_pnp_count = 0
    total_queries_evaluated = 0

    # Load preprocessed SfM data
    sfm_model_path = args.sfm_dir / scene / "sfm_superpoint+lightglue"
    if not sfm_model_path.exists():
        logger.warning(f"SfM Model missing for {scene}. Skipping.")
        return t_errors, r_errors, failed_pnp_count, total_queries_evaluated
        
    reconstruction = pycolmap.Reconstruction(sfm_model_path)
    cameras, images, _ = rw.read_model(sfm_model_path, ext=".bin")
    
    with open(args.covisibility_dir / scene / "covisibility_results.pkl", "rb") as f:
        covis_dict = pickle.load(f)

    if is_megadepth:
        # MegaDepth GT and paths
        query_cams = load_query_cams(args.query_dir / scene / "query_image_cameras.txt")
        query_names_file = args.query_dir / scene / "query_image_names_clean.txt"
        img_dir_base = args.dataset / scene / "images"
    else:
        # Cambridge GT and paths
        gt_model_path = args.query_dir / scene / "empty_all"
        cameras_gt, images_gt, _ = rw.read_model(gt_model_path, ext=".txt")
        query_cams = {}
        for img in images_gt.values():
            cam = cameras_gt[img.camera_id]
            query_cams[img.name] = {
                "qvec": img.qvec,
                "tvec": img.tvec,
                "intrinsics": {
                    "model": cam.model,
                    "width": cam.width,
                    "height": cam.height,
                    "params": cam.params
                }
            }
        query_names_file = args.query_dir / scene / "list_query.txt"
        img_dir_base = args.dataset / scene
        
    with open(query_names_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]

    active_pair_file = args.covisibility_dir / scene / "most_similar_pairs.txt"

    for query_name in tqdm(queries, desc=f"Evaluating {scene}"):
        if query_name not in covis_dict or query_name not in query_cams:
            continue
            
        total_queries_evaluated += 1

        # Get the most similar reference image
        primary_ref = get_most_similar_ref(query_name, active_pair_file)
        
        if not primary_ref:
            failed_pnp_count += 1
            t_errors.append(np.inf)
            r_errors.append(np.inf)
            continue

        # Prepare HLOC references from covisibility
        top_refs = []
        if args.method == "HLOC":
            valid_image_ids = covis_dict[query_name].get('unique_images', set())
            top_refs = [images[img_id].name for img_id in valid_image_ids if img_id in images]
            
            if primary_ref in top_refs:
                top_refs.remove(primary_ref)
            top_refs.insert(0, primary_ref)

        ref_image_obj = next((img for img in images.values() if img.name == primary_ref), None)
        ref_R = qvec2rotmat(ref_image_obj.qvec)
        ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))
        ref_cam_obj = cameras[ref_image_obj.camera_id]

        camera = query_cams[query_name]
        
        # Load query images and features
        img_query_path = img_dir_base / query_name
        img_query_pil = Image.open(img_query_path)
        q_img_size = np.array([img_query_pil.width, img_query_pil.height])
        
        features_path = args.sfm_dir / scene / "feats-superpoint-n2048.h5"
        
        with h5py.File(features_path, "r") as f:
            if query_name not in f: continue
            q_kpts = f[query_name]["keypoints"][:]
            q_desc = f[query_name]["descriptors"][:]

        visible_p3d = covis_dict[query_name]["unique_points"]
        p3d_desc, p3d_kpts = [], []
        p3d_indices_map = {} 

        with h5py.File(args.covisibility_dir / scene / "points3D_feats_cache.h5", "r") as f:
            idx_counter = 0
            for pid in visible_p3d:
                if str(pid) in f and int(pid) in reconstruction.points3D:
                    p3d_desc.append(f[str(pid)]["descriptors"][:].reshape(256))
                    p3d_kpts.append(f[str(pid)]["keypoints"][:].reshape(3))
                    p3d_indices_map[int(pid)] = idx_counter
                    idx_counter += 1
        
        if not p3d_kpts:
            failed_pnp_count += 1
            t_errors.append(np.inf)
            r_errors.append(np.inf)
            continue
            
        p3d_desc = np.vstack(p3d_desc).T 
        p3d_kpts = np.vstack(p3d_kpts) 

        # Prepare reference data strictly from the covisibility `top_refs`
        ref_data_list = []
        if args.method == "HLOC":
            with h5py.File(features_path, "r") as f:
                for r_name in top_refs:
                    if r_name in f:
                        # Load reference images and features
                        r_img_path = img_dir_base / r_name
                        r_img_pil = Image.open(r_img_path)
                        r_img_obj = next((img for img in images.values() if img.name == r_name), None)
                        
                        if r_img_obj is not None:
                            ref_data_list.append({
                                "kpts": f[r_name]["keypoints"][:],
                                "desc": f[r_name]["descriptors"][:],
                                "img_size": [r_img_pil.width, r_img_pil.height],
                                "p3d_ids": r_img_obj.point3D_ids,
                                "name": r_name,
                                "image_id": r_img_obj.id
                            })

        # Matching
        if args.method == "HLOC":
            pred_matches0 = compute_hloc_baseline(matchers['baseline'], q_kpts, q_desc, q_img_size, ref_data_list, p3d_indices_map, device)
        elif args.method == "NN":
            pred_matches0 = compute_nn_baseline(q_desc, p3d_desc, device)
        elif args.method == "RR":
            pred_matches0, _, _, _, _ = compute_rr_baseline(matchers['baseline'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device)
        elif args.method == "RN":
            pred_matches0, _, _, _, _ = compute_rn_baseline(matchers['baseline'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device)
        elif args.method == "PR":
            pred_matches0, _, _, _, _ = compute_pr_baseline(matchers['baseline'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, ref_cam_obj, device)
        elif args.method == "TRAIN":
            pred_matches0, _ = compute_trained_lightglu3d(matchers['lightglu3d'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)
        elif args.method == "ADAPT":
            pred_matches0, _ = compute_trained_lightglu3d(matchers['adapt'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)

        # Pose estimation
        valid_mask = pred_matches0 > -1
        matched_2d = q_kpts[valid_mask] + 0.5
        matched_3d = p3d_kpts[pred_matches0[valid_mask]]

        pose_res = evaluate_pose(matched_2d, matched_3d, camera, q_img_size, args.max_error)
    
        if pose_res is not None:
            t_err = pose_res["t_error"]
            # Apply val scaler here
            if is_megadepth:
                if scene == "0015":
                    t_err *= 16.0
                elif scene == "0022":
                    t_err *= 8.0
            t_errors.append(t_err)
            r_errors.append(pose_res["r_error_deg"])
        else:
            failed_pnp_count += 1
            t_errors.append(np.inf)
            r_errors.append(np.inf)

    return t_errors, r_errors, failed_pnp_count, total_queries_evaluated

def log_metrics(t_errors, r_errors, failed_pnp_count, total_queries, method_label):
    t_errs = np.array(t_errors)
    r_errs = np.array(r_errors)
    acc_strict = np.mean((t_errs <= 0.25) & (r_errs <= 2.0)) * 100
    acc_medium = np.mean((t_errs <= 0.50) & (r_errs <= 5.0)) * 100
    acc_loose  = np.mean((t_errs <= 5.00) & (r_errs <= 10.0)) * 100

    logger.info("="*40)
    logger.info(f"FINAL LOCALIZATION METRICS: {method_label}")
    logger.info("="*40)
    logger.info(f"Total Queries Evaluated:  {total_queries}")
    logger.info(f"Failed PnP Estimations:   {failed_pnp_count} images")
    logger.info(f"Median Trans Error:       {np.median(t_errs):.4f} m")
    logger.info(f"Median Rot Error:         {np.median(r_errs):.4f} deg")
    logger.info("-"*40)
    logger.info("Aachen Format (Translation & Rotation Limits):")
    logger.info(f"Strict (0.25m, 2°):     {acc_strict:.2f}%")
    logger.info(f"Medium (0.50m, 5°):     {acc_medium:.2f}%")
    logger.info(f"Loose  (5.00m, 10°):    {acc_loose:.2f}%")
    logger.info("="*40)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pose Accuracy over MegaDepth Scenes")
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--covisibility_dir', type=Path, required=True)
    parser.add_argument('--query_dir', type=Path, required=True)
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--scene_list', type=Path, required=True, help="Path to txt file with list of scenes")
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
        
    logger.info(f"Starting Pose Evaluation for {len(scenes)} MegaDepth scenes using {method} (RANSAC max_error: {args.max_error}px)")

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
        
        t_errs, r_errs, fails, total = process_scene(scene, args, matchers, device, is_megadepth=True)
        
        all_t_errors.extend(t_errs)
        all_r_errors.extend(r_errs)
        global_failed_pnp_count += fails
        global_total_queries += total

    # Print metrics
    log_metrics(all_t_errors, all_r_errors, global_failed_pnp_count, global_total_queries, method)

if __name__ == "__main__":
    main()