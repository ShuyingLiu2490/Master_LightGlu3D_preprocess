import argparse
import logging
import pickle
import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat
from ground_truth.generate_gt_pairs_by_scene import load_query_cams, compute_ground_truth_matches_soft
from lightglue import LightGlue
from baseline.rr_baseline import load_similar_pairs, compute_precision_recall

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_pr_baseline_change(matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, q_camera, device):

    # Rotate 3D points into reference pose (extrinsics)
    R = ref_pose_matrix[:, :3]
    t = ref_pose_matrix[:, 3]
    p3d_cam = (R @ p3d_kpts.T).T + t
    
    # Perspective projection
    Z = np.maximum(p3d_cam[:, 2], 1e-5)
    x_norm = p3d_cam[:, 0] / Z
    y_norm = p3d_cam[:, 1] / Z

    # Project using query intrinsics
    intrinsics = q_camera["intrinsics"]
    params = intrinsics["params"]
    camera_model = str(intrinsics["model"])
    
    if camera_model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "0", "1", "4", "8"]: 
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]
    else: 
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]

    p3d_proj_x = x_norm * fx + cx
    p3d_proj_y = y_norm * fy + cy

    # Push points behind the camera safely off-screen
    invalid_depth = p3d_cam[:, 2] <= 0
    p3d_proj_x[invalid_depth] = -9999.0
    p3d_proj_y[invalid_depth] = -9999.0

    p3d_proj_kpts = np.column_stack((p3d_proj_x, p3d_proj_y))

    proj_w, proj_h = intrinsics["width"], intrinsics["height"]

    # Format in LightGlue
    feats0 = {
        "keypoints": torch.from_numpy(q_kpts).float().unsqueeze(0).to(device),
        "descriptors": torch.from_numpy(q_desc.T).float().unsqueeze(0).to(device),
        "image_size": torch.from_numpy(np.array(q_img_size)).float().unsqueeze(0).to(device)
    }
    
    feats1 = {
        "keypoints": torch.from_numpy(p3d_proj_kpts).float().unsqueeze(0).to(device),
        "descriptors": torch.from_numpy(p3d_desc.T).float().unsqueeze(0).to(device),
        "image_size": torch.tensor([[proj_w, proj_h]]).float().to(device)
    }
    
    # Predict matches
    with torch.no_grad():
        res = matcher({"image0": feats0, "image1": feats1})
        
    matches = res["matches"][0].cpu().numpy()
    
    pred_matches0 = np.full(len(q_kpts), -1)
    if len(matches) > 0:
        pred_matches0[matches[:, 0]] = matches[:, 1]
        
    return pred_matches0, res, p3d_proj_kpts, proj_w, proj_h


def main():
    parser = argparse.ArgumentParser(description="Evaluate Changed PR Baseline across all test scenes")
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--covisibility_dir', type=Path, required=True)
    parser.add_argument('--query_dir', type=Path, required=True)
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--depth_dir', type=Path, required=True)
    parser.add_argument('--scene_list', type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize LightGlue
    logger.info("Initializing LightGlue...")
    matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)

    # Load test scene list
    with open(args.scene_list, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]

    overall_precisions = []
    overall_recalls = []

    # Process each scene
    for scene in scenes:
        logger.info(f"Starting Changed PR Baseline Evaluation for Scene {scene}...")
        
        # Load query
        query_names_file = args.query_dir / scene / "query_image_names_clean.txt"
        with open(query_names_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        pair_dict = load_similar_pairs(args.covisibility_dir / scene / "most_similar_pairs.txt")

        _, sfm_images, _ = rw.read_model(args.sfm_dir / scene / "sfm_superpoint+lightglue", ext=".bin")
        query_cams = load_query_cams(args.query_dir / scene / "query_image_cameras.txt")
        
        # Load covisibility
        with open(args.covisibility_dir / scene / "covisibility_results.pkl", "rb") as f:
            covis_dict = pickle.load(f)
 
        scene_precisions = []
        scene_recalls = []

        q_feats_path = args.sfm_dir / scene / "feats-superpoint-n2048.h5"
        p3d_feats_path = args.covisibility_dir / scene / "points3D_feats_cache.h5"
        
        with h5py.File(q_feats_path, "r") as q_feats_h5, h5py.File(p3d_feats_path, "r") as p3d_feats_h5:
            
            # Iterate through all queries in the scene
            for query_name in tqdm(queries, desc=f"Evaluating Queries in {scene}"):
                
                ref_name = pair_dict.get(query_name)
                if not ref_name:
                    continue 
                
                if query_name not in covis_dict:
                    continue
                visible_p3d = covis_dict[query_name]["unique_points"]
                if len(visible_p3d) == 0:
                    continue

                # Get query features
                q_kpts = q_feats_h5[query_name]["keypoints"][:]
                q_desc = q_feats_h5[query_name]["descriptors"][:]

                # Get 3D features
                p3d_desc, p3d_kpts = [], []
                for pid in visible_p3d:
                    pid_str = str(pid)
                    if pid_str in p3d_feats_h5:
                        p3d_desc.append(p3d_feats_h5[pid_str]["descriptors"][:].reshape(256))
                        p3d_kpts.append(p3d_feats_h5[pid_str]["keypoints"][:].reshape(3))
                        
                if len(p3d_kpts) == 0:
                    continue
                    
                p3d_desc = np.vstack(p3d_desc).T 
                p3d_kpts = np.vstack(p3d_kpts)   

                # Compute ground truth
                q_camera = query_cams[query_name]
                q_img_size = [q_camera["intrinsics"]["width"], q_camera["intrinsics"]["height"]]
                
                depth_file = args.depth_dir / scene / f"{Path(query_name).stem}.h5"
                if not depth_file.exists():
                    continue
                with h5py.File(depth_file, 'r') as f_depth:
                    depth_map = f_depth['depth'][:]
                    
                gt_matches0, _ = compute_ground_truth_matches_soft({"keypoints": q_kpts}, {"keypoints": p3d_kpts}, q_camera, depth_map)

                # Reference camera pose
                ref_image_obj = next((img for img in sfm_images.values() if img.name == ref_name), None)
                if not ref_image_obj:
                    continue
                ref_R = qvec2rotmat(ref_image_obj.qvec)
                ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))

                # Run changed PR baseline
                pr_matches0, _, _, _, _ = compute_pr_baseline_change(matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, q_camera, device)

                # Metrics
                precision, recall, _, _, _ = compute_precision_recall(pr_matches0, gt_matches0)
                if precision is not None:
                    scene_precisions.append(precision)
                if recall is not None:
                    scene_recalls.append(recall)

        # Summary
        avg_scene_precision = np.mean(scene_precisions) if scene_precisions else 0
        avg_scene_recall = np.mean(scene_recalls) if scene_recalls else 0
        
        logger.info("="*40)
        logger.info(f"Changed PR Baseline Results for Scene: {scene}")
        logger.info(f"Evaluated Queries: {len(scene_precisions)}")
        logger.info(f"Average Precision: {avg_scene_precision:.4f}")
        logger.info(f"Average Recall:    {avg_scene_recall:.4f}")
        logger.info("="*40)

        overall_precisions.extend(scene_precisions)
        overall_recalls.extend(scene_recalls)

    logger.info("="*40)
    logger.info("OVERALL CHANGED PR BASELINE RESULTS")
    logger.info(f"Total Scenes Evaluated:  {len(scenes)}")
    logger.info(f"Total Queries Evaluated: {len(overall_precisions)}")
    logger.info(f"Average Precision: {np.mean(overall_precisions):.4f}")
    logger.info(f"Average Recall:    {np.mean(overall_recalls):.4f}")
    logger.info("="*40)


if __name__ == "__main__":
    main()