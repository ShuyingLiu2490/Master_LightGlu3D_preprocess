# For baseline: NN(Nearest Neighbour), RR(Rotate+Remove_coord), 
#               RN(Rotate+Normalize), PR(Project to Reference),
#               PRC(PR change),
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
import pycolmap
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat, get_most_similar_ref
from tqdm import tqdm
from lightglue import LightGlue
from baseline.pr_baseline import compute_pr_baseline
from baseline.pr_baseline_change import compute_pr_baseline_change
from baseline.rr_baseline import compute_rr_baseline
from baseline.rn_baseline import compute_rn_baseline
from baseline.nn_baseline import compute_nn_baseline
from baseline.trained_matcher import load_trained_lightglu3d, load_trained_adapt, compute_trained_lightglu3d, compute_trained_lightglu3d_greedy_dynamic, compute_trained_lightglu3d_dynamic
from .pose_estimation import compute_hloc_baseline

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_reference_poses(filepath):
    poses = {}
    if filepath and filepath.exists():
        logger.info(f"Loading reference poses from {filepath}...")
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Format: name qw qx qy qz tx ty tz
                if len(parts) == 8:
                    name = Path(parts[0]).name 
                    # Store qvec as [W, X, Y, Z]
                    qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                    tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
                    poses[name] = {'qvec': qvec, 'tvec': tvec}
    else:
        if filepath:
            logger.warning(f"Reference file not found at {filepath}. Debug distance calculation will be skipped.")
    return poses

def parse_aachen_cameras(query_dir):
    cameras = {}
    query_files = [query_dir / "day_time_queries_with_intrinsics.txt",
                   query_dir / "night_time_queries_with_intrinsics.txt"]

    for query_file in query_files:
        if query_file.exists():
            with open(query_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        parts = line.strip().split()
                        img_name = parts[0]
                        model_name = parts[1]
                        width = int(parts[2])
                        height = int(parts[3])
                        params = np.array(parts[4:], dtype=float)
                        
                        cameras[img_name] = pycolmap.Camera(
                            model=model_name,
                            width=width,
                            height=height,
                            params=params
                        )
    return cameras

def estimate_pose_blind(matched_2d, matched_3d, camera, q_img_size, max_error, is_debug=False): 
    if len(matched_2d) < 4:
        return None

    orig_w = camera.width
    orig_h = camera.height
    new_w, new_h = q_img_size[0], q_img_size[1]
    
    params = np.array(camera.params, dtype=float)
    
    if orig_w != new_w or orig_h != new_h:
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        model_name = getattr(camera, 'model_name', camera.model.name if hasattr(camera.model, 'name') else str(camera.model))
        
        if model_name in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL_FISHEYE", "SIMPLE_RADIAL_FISHEYE"]:
            params[0] *= scale_x # f
            params[1] *= scale_x # cx
            params[2] *= scale_y # cy
        elif model_name in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "RADIAL"]:
            params[0] *= scale_x # fx
            params[1] *= scale_y # fy
            params[2] *= scale_x # cx
            params[3] *= scale_y # cy
        else:
            logger.warning(f"Unrecognized camera model {model_name}. Scaling might be incorrect!")

    colmap_cam = pycolmap.Camera(
        model=camera.model,
        width=int(new_w),
        height=int(new_h),
        params=params
    )

    estimation_options = {"ransac": {"max_error": max_error}}
    refinement_options = {"refine_focal_length": False, "refine_extra_params": False}

    ret = pycolmap.estimate_and_refine_absolute_pose(
        matched_2d, matched_3d, colmap_cam, estimation_options, refinement_options
    )

    if ret is None or not ret.get("is_valid", True):
        return None

    if isinstance(ret, dict):
        qvec = ret["cam_from_world"].rotation.quat # PyCOLMAP returns [X, Y, Z, W]
        tvec = ret["cam_from_world"].translation
        inliers = ret["inlier_mask"]
    else:
        qvec = ret.qvec
        tvec = ret.tvec
        inliers = ret.inlier_mask

    return {
        "qvec": qvec,
        "tvec": tvec,
        "inliers": inliers
    }

def write_benchmark_file(poses, output_file):
    logger.info(f"Writing {len(poses)} estimated poses to {output_file}")
    with open(output_file, 'w') as f:
        for img_name, pose in poses.items():
            benchmark_name = Path(img_name).name
            q = pose['qvec']
            t = pose['tvec']
            # Reorder PyCOLMAP's [X, Y, Z, W] into benchmark's [W, X, Y, Z]
            f.write(f"{benchmark_name} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]}\n")

def process_aachen(args, device):
    method = args.method
    logger.info(f"Starting Blind Pose Estimation for Aachen v1.1 using {method} (RANSAC: {args.max_error}px)")

    # Load dynamic HLOC reference poses
    reference_poses = load_reference_poses(args.hloc_reference)

    # Initialize matchers
    if method in ["RR", "RN", "PR", "PRC", "HLOC"]:
        baseline_matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    elif method == "TRAIN":
        if args.checkpoint is None: raise ValueError("--checkpoint must be provided.")
        lightglu3d_matcher = load_trained_lightglu3d(args.checkpoint, device, filter_threshold=0.015)
    elif method == "ADAPT":
        if args.checkpoint is None: raise ValueError("--checkpoint must be provided.")
        lightglu3d_adapt_matcher = load_trained_adapt(args.checkpoint, device)

    # Load models and data
    sfm_model_path = args.sfm_dir / "sfm_superpoint+lightglue"
    if not sfm_model_path.exists():
        raise FileNotFoundError(f"SfM model not found at {sfm_model_path}")
        
    reconstruction = pycolmap.Reconstruction(sfm_model_path)
    cameras, images, _ = rw.read_model(sfm_model_path, ext=".bin")
    
    with open(args.covisibility_dir / "covisibility_results.pkl", "rb") as f:
        covis_dict = pickle.load(f)
        
    aachen_query_cams = parse_aachen_cameras(args.query_dir)
    
    clean_query_file = args.covisibility_dir / "clean_aachen_queries.txt"
    with open(clean_query_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]

    active_pair_file = args.covisibility_dir / "most_similar_pairs.txt"
    
    estimated_poses = {}
    failed_pnp_count = 0

    # Preload features
    features_path = args.sfm_dir / "feats-superpoint-n2048.h5"
    p3d_feats_path = args.covisibility_dir / "points3D_feats_cache.h5"
    features_h5 = h5py.File(features_path, "r")
    p3d_feats_h5 = h5py.File(p3d_feats_path, "r")

    for query_name in tqdm(queries, desc=f"Evaluating Aachen"):
        
        is_debug_target = "IMG_20140520_182846.jpg" in query_name
        if is_debug_target:
            logger.info(f"[DEBUG] TRIGGERED FOR: {query_name}")

        if query_name not in covis_dict:
            failed_pnp_count += 1
            continue
            
        primary_ref = get_most_similar_ref(query_name, active_pair_file)
        if not primary_ref or query_name not in aachen_query_cams:
            failed_pnp_count += 1
            continue

        top_refs = []
        if method == "HLOC":
            valid_image_ids = covis_dict[query_name].get('unique_images', set())
            top_refs = [images[img_id].name for img_id in valid_image_ids if img_id in images]
            if primary_ref in top_refs:
                top_refs.remove(primary_ref)
            top_refs.insert(0, primary_ref)

        ref_image_obj = next((img for img in images.values() if img.name == primary_ref), None)
        ref_R = qvec2rotmat(ref_image_obj.qvec)
        ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))
        ref_cam_obj = cameras[ref_image_obj.camera_id]

        camera = aachen_query_cams[query_name]
        
        # Load query features
        if query_name not in features_h5: 
            failed_pnp_count += 1
            continue
        q_kpts = features_h5[query_name]["keypoints"][:]
        q_desc = features_h5[query_name]["descriptors"][:]
        q_img_size = np.array(features_h5[query_name]["image_size"][:])

        # Load covisible features
        visible_p3d = covis_dict[query_name]["unique_points"]
        p3d_desc, p3d_kpts, p3d_xyz = [], [], []
        p3d_indices_map = {} 

        idx_counter = 0
        for pid in visible_p3d:
            pid_str = str(pid)
            if pid_str in p3d_feats_h5 and int(pid) in reconstruction.points3D:
                p3d_desc.append(p3d_feats_h5[pid_str]["descriptors"][:].reshape(256))
                p3d_kpts.append(p3d_feats_h5[pid_str]["keypoints"][:].reshape(3))
                
                # Pull the exact XYZ coordinates straight from COLMAP
                p3d_xyz.append(reconstruction.points3D[int(pid)].xyz) 
                
                p3d_indices_map[int(pid)] = idx_counter
                idx_counter += 1
        
        if not p3d_kpts:
            failed_pnp_count += 1
            continue
            
        p3d_desc = np.vstack(p3d_desc).T 
        p3d_kpts = np.vstack(p3d_kpts) 
        p3d_xyz = np.vstack(p3d_xyz) # Stack the XYZ coordinates

        ref_data_list = []
        if method == "HLOC":
            with h5py.File(features_path, "r") as f:
                for r_name in top_refs:
                    if r_name in f:
                        r_img_obj = next((img for img in images.values() if img.name == r_name), None)
                        if r_img_obj is not None:
                            r_img_size = f[r_name]["image_size"][:]
                            ref_data_list.append({
                                "kpts": f[r_name]["keypoints"][:], 
                                "desc": f[r_name]["descriptors"][:],
                                "img_size": r_img_size,
                                "p3d_ids": r_img_obj.point3D_ids,
                                "name": r_name, 
                                "image_id": r_img_obj.id
                            })

        # Matching
        if method == "HLOC":
            pred_matches0 = compute_hloc_baseline(baseline_matcher, q_kpts, q_desc, q_img_size, ref_data_list, p3d_indices_map, device)
        elif method == "NN":
            pred_matches0 = compute_nn_baseline(q_desc, p3d_desc, device)
        elif method == "RR":
            pred_matches0, _, _, _, _ = compute_rr_baseline(baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device)
        elif method == "RN":
            pred_matches0, _, _, _, _ = compute_rn_baseline(baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device)
        elif method == "PR":
            pred_matches0, _, _, _, _ = compute_pr_baseline(baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, ref_cam_obj, device)
        elif method == "PRC":
            q_camera_dict = {
                "intrinsics": {
                    "model": getattr(camera, 'model_name', getattr(camera.model, 'name', str(camera.model))),
                    "width": camera.width,
                    "height": camera.height,
                    "params": camera.params
                }
            }
            pred_matches0, _, _, _, _ = compute_pr_baseline_change(baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, q_camera_dict, device)
        elif method == "TRAIN":
            if args.greedy_or_mutual == 'greedy':
                pred_matches0, _ = compute_trained_lightglu3d_greedy_dynamic(lightglu3d_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device, min_matches=args.min_matches)
            else:  
                pred_matches0, _ = compute_trained_lightglu3d_dynamic(lightglu3d_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device, min_matches=args.min_matches)
        elif method == "ADAPT":
            pred_matches0, _ = compute_trained_lightglu3d(lightglu3d_adapt_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)

        # Pose estimation
        valid_mask = pred_matches0 > -1
        matched_2d = q_kpts[valid_mask] + 0.5
        matched_3d = p3d_xyz[pred_matches0[valid_mask]] # Use the true PyCOLMAP XYZ coordinates

        if is_debug_target:
            logger.info(f"[DEBUG] Feeding {len(matched_2d)} matches into PyCOLMAP")

        # Pose estimation
        pose_res = estimate_pose_blind(matched_2d, matched_3d, camera, q_img_size, args.max_error, is_debug=is_debug_target)
    
        if pose_res is not None:
            estimated_poses[query_name] = pose_res
            
            if is_debug_target:
                t = pose_res['tvec']
                q = pose_res['qvec'] # PyCOLMAP returns [X, Y, Z, W]
                num_inliers = len(pose_res['inliers']) if isinstance(pose_res['inliers'], list) else np.sum(pose_res['inliers'])
                
                logger.info(f"[DEBUG] PnP Success! Inliers: {num_inliers}")

                # Check against the dynamic HLOC reference
                benchmark_name = Path(query_name).name
                expected_pose = reference_poses.get(benchmark_name)
                
                if expected_pose is not None:
                    expected_t = expected_pose['tvec']
                    expected_q = expected_pose['qvec'] # HLOC reference is [W, X, Y, Z]
                    
                    # Distance (m)
                    dist = np.linalg.norm(t - expected_t)
                    
                    # Angle (Deg)
                    q_aligned = np.array([q[3], q[0], q[1], q[2]])
                    dot_product = np.abs(np.dot(q_aligned, expected_q))
                    dot_product = np.clip(dot_product, 0.0, 1.0)
                    angle_diff = np.rad2deg(2 * np.arccos(dot_product))
                    
                    logger.info(f"[DEBUG] HLOC Reference Tvec: {expected_t}")
                    logger.info(f"[DEBUG] Euclidean Distance:  {dist:.2f} meters")
                    logger.info(f"[DEBUG] Angular Difference:  {angle_diff:.2f} degrees")
                    
                    if dist < 10.0 and angle_diff < 10.0:
                        logger.info("[DEBUG] SUCCESS! Coordinates and rotations are acceptable.")
                    else:
                        logger.info("[DEBUG] WARNING: Significant discrepancy found against HLOC reference!")
                else:
                    logger.info(f"[DEBUG] WARNING: Could not find '{benchmark_name}' in the provided HLOC reference file.")
        else:
            failed_pnp_count += 1
            if is_debug_target:
                logger.error("[DEBUG] PyCOLMAP failed to find a valid pose for this image!")

    success_count = len(estimated_poses)

    # Output
    logger.info("="*40)
    logger.info(f"Aachen Evaluation Completed: {method}")
    logger.info("="*40)
    logger.info(f"Total Queries:      {len(queries)}")
    logger.info(f"Successfully PnP:   {success_count}")
    logger.info(f"Failed PnP:         {failed_pnp_count}")
    logger.info("="*40)

    # output_file = args.outputs / f"Aachen_v1_1_eval_{method}_{args.greedy_or_mutual}_{args.min_matches}_{args.name_extra}.txt"
    output_file = args.outputs / f"Aachen_v1_1_eval_{method}.txt"
    write_benchmark_file(estimated_poses, output_file)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pose Accuracy on Aachen v1.1 Benchmark")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Aachen root")
    parser.add_argument('--covisibility_dir', type=Path, required=True, help="Path to output directory (where clean_queries and results.pkl live)")
    parser.add_argument('--image_dir', type=Path, required=True, help="Path to unlocked images_upright dir")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to the sfm directory")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to original queries directory")
    parser.add_argument('--outputs', type=Path, required=True, help="Where to save the final Benchmark .txt file")
    parser.add_argument('--method', type=str, required=True, choices=['NN', 'RR', 'RN', 'PR', 'PRC', 'TRAIN', 'ADAPT', 'HLOC'], help="Matching method")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to trained weights")
    # parser.add_argument('--name_extra', type=str, default="", help="Extra name info to distinguish output files")
    parser.add_argument('--max_error', type=float, default=12.0, help="RANSAC Reprojection Error Threshold")
    parser.add_argument('--hloc_reference', type=Path, default=None, help="Path to HLOC generated txt file for coordinate verification")
    parser.add_argument('--greedy_or_mutual', type=str, default='greedy', choices=['greedy', 'mutual'], help="Whether to use greedy or mutual filtering for the dynamic thresholding in the TRAIN method")
    parser.add_argument('--min_matches', type=int, default=800, help="Minimum matches for dynamic LightGlue")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        process_aachen(args, device)
    except Exception as e:
        logger.error(f"Failed to process Aachen Pose Estimation: {e}", exc_info=True)

if __name__ == "__main__":
    main()