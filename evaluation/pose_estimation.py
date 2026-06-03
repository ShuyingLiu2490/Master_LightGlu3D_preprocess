import argparse
import logging
import pickle
import numpy as np
import torch
import h5py
from pathlib import Path
from PIL import Image
import pycolmap
from tqdm import tqdm
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat, get_most_similar_ref
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_cambridge(scene, args, matchers, device):
    t_errors, r_errors = [], []
    failed_pnp_count = 0
    total_queries_evaluated = 0

    sfm_model_path = args.sfm_dir / scene / "sfm_superpoint+lightglue"
    if not sfm_model_path.exists():
        logger.warning(f"SfM Model missing for {scene}. Skipping.")
        return t_errors, r_errors, failed_pnp_count, total_queries_evaluated
        
    reconstruction = pycolmap.Reconstruction(sfm_model_path)
    cameras, images, _ = rw.read_model(sfm_model_path, ext=".bin")
    
    with open(args.covisibility_dir / scene / "covisibility_results.pkl", "rb") as f:
        covis_dict = pickle.load(f)

    # Cambridge GT and paths
    gt_model_path = args.query_dir / scene / "empty_all"
    cameras_gt, images_gt, _ = rw.read_model(gt_model_path, ext=".txt")
    query_cams = {}
    for img in images_gt.values():
        cam = cameras_gt[img.camera_id]
        query_cams[img.name] = {
            "qvec": img.qvec, "tvec": img.tvec,
            "intrinsics": {"model": cam.model, "width": cam.width, "height": cam.height, "params": cam.params}
        }
    
    query_names_file = args.query_dir / scene / "list_query.txt"
    img_dir_base = args.dataset / scene
        
    with open(query_names_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]

    active_pair_file = args.covisibility_dir / scene / "most_similar_pairs.txt"

    for query_name in tqdm(queries, desc=f"Evaluating {scene}"):
        if query_name not in covis_dict or query_name not in query_cams: continue
            
        total_queries_evaluated += 1
        primary_ref = get_most_similar_ref(query_name, active_pair_file)
        
        if not primary_ref:
            failed_pnp_count += 1
            t_errors.append(np.inf); r_errors.append(np.inf)
            continue

        top_refs = []
        if args.method == "HLOC":
            valid_image_ids = covis_dict[query_name].get('unique_images', set())
            top_refs = [images[img_id].name for img_id in valid_image_ids if img_id in images]
            if primary_ref in top_refs: top_refs.remove(primary_ref)
            top_refs.insert(0, primary_ref)

        ref_image_obj = next((img for img in images.values() if img.name == primary_ref), None)
        ref_R = qvec2rotmat(ref_image_obj.qvec)
        ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))
        
        camera = query_cams[query_name]
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
            t_errors.append(np.inf); r_errors.append(np.inf)
            continue
            
        p3d_desc = np.vstack(p3d_desc).T 
        p3d_kpts = np.vstack(p3d_kpts) 

        ref_data_list = []
        if args.method == "HLOC":
            with h5py.File(features_path, "r") as f:
                for r_name in top_refs:
                    if r_name in f:
                        r_img_path = img_dir_base / r_name
                        r_img_pil = Image.open(r_img_path)
                        r_img_obj = next((img for img in images.values() if img.name == r_name), None)
                        if r_img_obj is not None:
                            ref_data_list.append({
                                "kpts": f[r_name]["keypoints"][:], "desc": f[r_name]["descriptors"][:],
                                "img_size": [r_img_pil.width, r_img_pil.height],
                                "p3d_ids": r_img_obj.point3D_ids, "name": r_name, "image_id": r_img_obj.id
                            })

        # Matcher Routing
        if args.method == "HLOC":
            pred_matches0 = compute_hloc_baseline(matchers['baseline'], q_kpts, q_desc, q_img_size, ref_data_list, p3d_indices_map, device)
        elif args.method == "MNN":
            pred_matches0 = compute_nn_baseline(q_desc, p3d_desc, device)
        elif args.method == "PR":
            pred_matches0, _, _, _, _ = compute_pr_baseline(matchers['baseline'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, camera, device)
        elif args.method == "TRAIN":
            pred_matches0, _ = compute_trained_lightglu3d(matchers['lightglu3d'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)

        valid_mask = pred_matches0 > -1
        matched_2d = q_kpts[valid_mask] + 0.5
        matched_3d = p3d_kpts[pred_matches0[valid_mask]]

        pose_res = evaluate_pose(matched_2d, matched_3d, camera, q_img_size, args.max_error)
    
        if pose_res is not None:
            t_errors.append(pose_res["t_error"])
            r_errors.append(pose_res["r_error_deg"])
        else:
            failed_pnp_count += 1
            t_errors.append(np.inf); r_errors.append(np.inf)

    return t_errors, r_errors, failed_pnp_count, total_queries_evaluated

def process_aachen(args, matchers, device):
    logger.info(f"Starting Blind Pose Estimation for Aachen v1.1 using {args.method} (RANSAC: {args.max_error}px)")

    sfm_model_path = args.sfm_dir / "sfm_superpoint+lightglue"
    reconstruction = pycolmap.Reconstruction(sfm_model_path)
    cameras, images, _ = rw.read_model(sfm_model_path, ext=".bin")
    
    with open(args.covisibility_dir / "covisibility_results.pkl", "rb") as f:
        covis_dict = pickle.load(f)
        
    aachen_query_cams = {}
    for query_file in [args.query_dir / "day_time_queries_with_intrinsics.txt", args.query_dir / "night_time_queries_with_intrinsics.txt"]:
        if query_file.exists():
            with open(query_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        parts = line.strip().split()
                        aachen_query_cams[parts[0]] = pycolmap.Camera(
                            model=parts[1], width=int(parts[2]), height=int(parts[3]), params=np.array(parts[4:], dtype=float)
                        )
    
    with open(args.covisibility_dir / "clean_aachen_queries.txt", 'r') as f:
        queries = [line.strip() for line in f if line.strip()]

    active_pair_file = args.covisibility_dir / "most_similar_pairs.txt"
    estimated_poses = {}
    failed_pnp_count = 0

    features_h5 = h5py.File(args.sfm_dir / "feats-superpoint-n2048.h5", "r")
    p3d_feats_h5 = h5py.File(args.covisibility_dir / "points3D_feats_cache.h5", "r")

    for query_name in tqdm(queries, desc=f"Evaluating Aachen"):
        if query_name not in covis_dict:
            failed_pnp_count += 1; continue
            
        primary_ref = get_most_similar_ref(query_name, active_pair_file)
        if not primary_ref or query_name not in aachen_query_cams:
            failed_pnp_count += 1; continue

        top_refs = []
        if args.method == "HLOC":
            valid_image_ids = covis_dict[query_name].get('unique_images', set())
            top_refs = [images[img_id].name for img_id in valid_image_ids if img_id in images]
            if primary_ref in top_refs: top_refs.remove(primary_ref)
            top_refs.insert(0, primary_ref)

        ref_image_obj = next((img for img in images.values() if img.name == primary_ref), None)
        ref_R = qvec2rotmat(ref_image_obj.qvec)
        ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))
        camera = aachen_query_cams[query_name]
        
        if query_name not in features_h5: 
            failed_pnp_count += 1; continue
        q_kpts = features_h5[query_name]["keypoints"][:]
        q_desc = features_h5[query_name]["descriptors"][:]
        q_img_size = np.array(features_h5[query_name]["image_size"][:])

        visible_p3d = covis_dict[query_name]["unique_points"]
        p3d_desc, p3d_kpts, p3d_xyz = [], [], []
        p3d_indices_map = {} 

        idx_counter = 0
        for pid in visible_p3d:
            if str(pid) in p3d_feats_h5 and int(pid) in reconstruction.points3D:
                p3d_desc.append(p3d_feats_h5[str(pid)]["descriptors"][:].reshape(256))
                p3d_kpts.append(p3d_feats_h5[str(pid)]["keypoints"][:].reshape(3))
                p3d_xyz.append(reconstruction.points3D[int(pid)].xyz) 
                p3d_indices_map[int(pid)] = idx_counter
                idx_counter += 1
        
        if not p3d_kpts:
            failed_pnp_count += 1; continue
            
        p3d_desc = np.vstack(p3d_desc).T 
        p3d_kpts = np.vstack(p3d_kpts) 
        p3d_xyz = np.vstack(p3d_xyz) 

        ref_data_list = []
        if args.method == "HLOC":
            for r_name in top_refs:
                if r_name in features_h5:
                    r_img_obj = next((img for img in images.values() if img.name == r_name), None)
                    if r_img_obj is not None:
                        ref_data_list.append({
                            "kpts": features_h5[r_name]["keypoints"][:], "desc": features_h5[r_name]["descriptors"][:],
                            "img_size": features_h5[r_name]["image_size"][:],
                            "p3d_ids": r_img_obj.point3D_ids, "name": r_name, "image_id": r_img_obj.id
                        })

        if args.method == "HLOC":
            pred_matches0 = compute_hloc_baseline(matchers['baseline'], q_kpts, q_desc, q_img_size, ref_data_list, p3d_indices_map, device)
        elif args.method == "MNN":
            pred_matches0 = compute_nn_baseline(q_desc, p3d_desc, device)
        elif args.method == "PR":
            q_cam_dict = {
                "intrinsics": {
                    "model": getattr(camera, 'model_name', getattr(camera.model, 'name', str(camera.model))),
                    "width": camera.width,
                    "height": camera.height,
                    "params": camera.params
                }
            }
            pred_matches0, _, _, _, _ = compute_pr_baseline(matchers['baseline'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, q_cam_dict, device)
        elif args.method == "TRAIN":
            if args.greedy_or_not:
                pred_matches0, _ = compute_trained_lightglu3d_greedy_dynamic(matchers['lightglu3d'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device, min_matches=args.min_matches)
            else:
                pred_matches0, _ = compute_trained_lightglu3d(matchers['lightglu3d'], q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)
                
        valid_mask = pred_matches0 > -1
        matched_2d = q_kpts[valid_mask] + 0.5
        matched_3d = p3d_xyz[pred_matches0[valid_mask]] 

        pose_res = estimate_pose_blind(matched_2d, matched_3d, camera, q_img_size, args.max_error)
    
        if pose_res is not None:
            estimated_poses[query_name] = pose_res
        else:
            failed_pnp_count += 1

    success_count = len(estimated_poses)
    logger.info("="*40)
    logger.info(f"Aachen Evaluation Completed: {args.method}")
    logger.info("="*40)
    logger.info(f"Total Queries:      {len(queries)}")
    logger.info(f"Successfully PnP:   {success_count}")
    logger.info(f"Failed PnP:         {failed_pnp_count}")
    logger.info("="*40)

    if args.outputs is not None:
        output_file = args.outputs / f"Aachen_v1_1_eval_{args.method}.txt"
        logger.info(f"Writing {len(estimated_poses)} estimated poses to {output_file}")
        with open(output_file, 'w') as f:
            for img_name, pose in estimated_poses.items():
                benchmark_name = Path(img_name).name
                q, t = pose['qvec'], pose['tvec']
                f.write(f"{benchmark_name} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]}\n")

def main():
    parser = argparse.ArgumentParser(description="Unified Pose Estimation for Cambridge and Aachen")
    parser.add_argument('--dataset_type', type=str, required=True, choices=['cambridge', 'aachen'], help="Which dataset logic to run")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to dataset root")
    parser.add_argument('--covisibility_dir', type=Path, required=True)
    parser.add_argument('--query_dir', type=Path, required=True)
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--scene_list', type=Path, default=None, help="Path to txt file with list of scenes (Cambridge)")
    parser.add_argument('--method', type=str, required=True, choices=['MNN', 'PR', 'TRAIN', 'HLOC'])
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to trained network weights (Required for TRAIN)")
    parser.add_argument('--max_error', type=float, default=12.0, help="RANSAC Reprojection Error Threshold (pixels)")
    parser.add_argument('--filter_threshold', type=float, default=0.1, help="Filter threshold for the trained LightGlu3D model")
    parser.add_argument('--outputs', type=Path, default=None, help="Output dir for Aachen benchmark txt file")
    parser.add_argument('--greedy_or_not', type=bool, default=False, help="Whether to use greedy matching for the trained LightGlu3D model (only for TRAIN method)")
    parser.add_argument('--min_matches', type=int, default=800, help="Minimum matches for dynamic LightGlue (Aachen)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = args.method

    # Initialize Matchers
    matchers = {}
    if method in ["PR", "HLOC"]:
        matchers['baseline'] = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    elif method == "TRAIN":
        if args.checkpoint is None: raise ValueError("--checkpoint must be provided for TRAIN method.")
        matchers['lightglu3d'] = load_trained_lightglu3d(args.checkpoint, device, filter_threshold=args.filter_threshold)

    # Route Execution
    if args.dataset_type == 'cambridge':
        if not args.scene_list: raise ValueError("--scene_list is required for Cambridge")
        with open(args.scene_list, 'r') as f:
            scenes = [line.strip() for line in f if line.strip()]
            
        logger.info(f"Starting Cambridge Evaluation over {len(scenes)} scenes.")
        all_t_errors, all_r_errors = [], []
        global_failed = 0; global_total = 0

        for scene in scenes:
            logger.info(f"Processing Scene: {scene}...")
            t_errs, r_errs, fails, total = process_cambridge(scene, args, matchers, device)
            
            if total > 0:
                log_metrics(t_errs, r_errs, fails, total, method_label=f"{method} - SCENE: {scene}")
                
            all_t_errors.extend(t_errs)
            all_r_errors.extend(r_errs)
            global_failed += fails
            global_total += total

        if global_total > 0:
            log_metrics(all_t_errors, all_r_errors, global_failed, global_total, method_label=f"{method} - CAMBRIDGE SUMMARY")

    elif args.dataset_type == 'aachen':
        if not args.outputs: raise ValueError("--outputs directory must be specified for Aachen to save benchmark results.")
        process_aachen(args, matchers, device)

if __name__ == "__main__":
    main()