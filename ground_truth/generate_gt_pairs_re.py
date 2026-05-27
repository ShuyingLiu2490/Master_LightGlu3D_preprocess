import h5py
from pathlib import Path
import pickle
import numpy as np
from utils.utils import qvec2rotmat
import torch
import torch.nn.functional as F

import argparse
import logging
from tqdm import tqdm
from scipy.spatial import cKDTree

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_query_decriptors(img_name, h5_path):
    
    query_feature_dict = {}
    with h5py.File(h5_path, "r") as f_h5:

        if img_name not in f_h5:
            logger.warning(f"{img_name} not found in {h5_path}")
            return query_feature_dict
            
        ds = f_h5[img_name]
        query_feature_dict["descriptors"] = ds["descriptors"][:]
        query_feature_dict["scores"] = ds["scores"][:]
        query_feature_dict["keypoints"] = ds["keypoints"][:]
    
    logger.info(f"Collected descriptors for query image {img_name}.")
        
    return query_feature_dict

def load_query_cams(query_pose_path):

    query_pose_dict = {}
    with open(query_pose_path, 'r') as f:
        for line in f:
            item = line.strip().split()
            name = item[0]

            qvec = list(map(float, item[1:5]))      # 4 numbers
            tvec = list(map(float, item[5:8]))      # 3 numbers

            camera_id = item[8]
            model = item[9]
            width = int(item[10])
            height = int(item[11])
            params = list(map(float, item[12:]))

            query_pose_dict[name] = {
                "qvec": qvec,
                "tvec": tvec,
                "intrinsics": {
                    "camera_id": camera_id,
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params
                }
            }
    return query_pose_dict

def extract_points3d_descriptors(points3d, h5_path):

    point3d_feature_dict = {}
    descriptors = []
    keypoints = []
    scores = []
    with h5py.File(h5_path, "r") as f_h5:
        for p3d_id in points3d:
            p3d_id_str = str(p3d_id)
            if p3d_id_str not in f_h5:
                logger.warning(f"3D Point {p3d_id_str} not found in cache.")
                continue
            ds = f_h5[p3d_id_str]
            descriptors.append(ds["descriptors"][:].reshape(1, 256))
            keypoints.append(ds["keypoints"][:].reshape(1, 3))
            scores.append(ds["scores"][:])

    if len(keypoints) == 0:
        logger.warning("No valid 3D points found.")
        return {}

    point3d_feature_dict["descriptors"] = np.vstack(descriptors)
    point3d_feature_dict["scores"] = scores
    point3d_feature_dict["keypoints"] = np.vstack(keypoints)

    logger.info(f"Collected descriptors for {len(keypoints)} 3D points.")

    return point3d_feature_dict


def sample_depth_bilinear(depth_map, u, v):
    """
    depth_map: (H, W)
    u, v: arrays (N,)
    return: depth values (N,)
    """

    H, W = depth_map.shape

    # normalize to [-1,1] for grid_sample
    u_norm = 2.0 * u / (W - 1) - 1.0
    v_norm = 2.0 * v / (H - 1) - 1.0

    grid = torch.from_numpy(
        np.stack([u_norm, v_norm], axis=-1)
    ).float().unsqueeze(0).unsqueeze(0)  # (1,1,N,2)

    depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0).unsqueeze(0)

    sampled = F.grid_sample(
        depth_tensor,
        grid,
        align_corners=True,
        mode='bilinear'
    )

    return sampled.squeeze().numpy()


def compute_ground_truth_matches(query_feats, p3d_feats, camera, 
                                 depth_map=None, reproj_thresh=3.0, 
                                 depth_rel_thresh=0.1):
    """
    return:
        matches0: (N2D,)
        matches1: (N3D,)
    """
    
    kpts2d = query_feats["keypoints"]      # (N2D, 2)
    pts3d = p3d_feats["keypoints"]         # (N3D, 3)

    N2D = kpts2d.shape[0]
    N3D = pts3d.shape[0]

    matches0 = -np.ones(N2D, dtype=int)
    matches1 = -np.ones(N3D, dtype=int)

    # pose
    R = qvec2rotmat(camera["qvec"])
    t = np.array(camera["tvec"]).reshape(3, 1)

    # intrinsics
    params = camera["intrinsics"]["params"]
    fx, fy, cx, cy = params[:4]# assuming PINHOLE: fx fy cx cy
    width = camera["intrinsics"]["width"]
    height = camera["intrinsics"]["height"]

    # project all 3D points
    X = pts3d.T  # (3, N3D)

    X_cam = R @ X + t  # (3, N3D)

    z = X_cam[2]
    valid = z > 0  # check depth > 0

    X_cam = X_cam[:, valid]
    z = z[valid]

    u = fx * (X_cam[0] / z) + cx
    v = fy * (X_cam[1] / z) + cy

    valid_proj = (
        (u >= 0) & (u < width) &
        (v >= 0) & (v < height)
    )

    u = u[valid_proj]
    v = v[valid_proj]
    z = z[valid_proj]

    valid_indices = np.where(valid)[0][valid_proj]

    # check depth consistency
    if depth_map is not None:

        depth_real = sample_depth_bilinear(depth_map, u, v)
        valid_depth = depth_real > 0
        rel_error = np.full_like(depth_real, np.inf)
        rel_error[valid_depth] = (
            np.abs(z[valid_depth] - depth_real[valid_depth])
            / depth_real[valid_depth]
        ).flatten()

        depth_mask = (rel_error <= depth_rel_thresh)

        u = u[depth_mask]
        v = v[depth_mask]
        z = z[depth_mask]
        valid_indices = valid_indices[depth_mask]

    projected = np.stack([u, v], axis=1)

    
    # Trying fast methods part:
    # Fisrt is the original search
    # find the nearest 2D keypoint
    # for idx3d, proj_pt in zip(valid_indices, projected):

    #     dists = np.linalg.norm(kpts2d - proj_pt, axis=1)
    #     min_idx = np.argmin(dists)

    #     if dists[min_idx] < reproj_thresh:

    #         if matches0[min_idx] == -1: # in case to rewrite, only register the first matched pair
    #             matches0[min_idx] = idx3d
    #             matches1[idx3d] = min_idx

    # Second is to use KDTree
    if len(projected) > 0 and N2D > 0:
        tree = cKDTree(kpts2d)
        
        # Query the tree for the nearest 2D keypoint to each projected 3D point
        # distance_upper_bound acts as an instant cutoff mask (reproj_thresh)
        dists, min_indices = tree.query(projected, distance_upper_bound=reproj_thresh)
        
        # Iterate over the valid results and assign matches
        for idx3d, min_idx, dist in zip(valid_indices, min_indices, dists):
            # cKDTree returns len(kpts2d) if no neighbor was found within the threshold
            if min_idx < N2D: 
                if matches0[min_idx] == -1:
                    matches0[min_idx] = idx3d
                    matches1[idx3d] = min_idx

    # Third is to use cidst
    # if len(projected) > 0 and N2D > 0:
    #     kpts_tensor = torch.from_numpy(kpts2d).float()
    #     proj_tensor = torch.from_numpy(projected).float()
        
    #     # Calculate the dense distance matrix
    #     dist_matrix = torch.cdist(kpts_tensor, proj_tensor)
        
    #     # Find the minimum distance along the 2D keypoint dimension
    #     min_dists, min_indices = torch.min(dist_matrix, dim=0)
        
    #     min_dists = min_dists.numpy()
    #     min_indices = min_indices.numpy()
        
    #     # Iterate over the valid results and assign matches
    #     for idx3d, min_idx, dist in zip(valid_indices, min_indices, min_dists):
    #         if dist < reproj_thresh:
    #             if matches0[min_idx] == -1:
    #                 matches0[min_idx] = idx3d
    #                 matches1[idx3d] = min_idx

    return matches0, matches1


def load_depth(depth_path):
    with h5py.File(depth_path, 'r') as f:
        depth = f['depth'][:]
    return depth


def generate_gt_for_query(query, feats_2d_path, feats_3d_path, query_cams, covisibility_dict, depth_path, args):
    # extract SP keypoints descriptors of the query
    query_feats = extract_query_decriptors(query, feats_2d_path)
    if not query_feats:
        return None

    # load pose and camera of the query
    camera = query_cams[query] # qvec, tvec...

    # extract covisibility results of the query
    visible_p3d = covisibility_dict[query]["unique_points"]
    
    # load keypoints and descriptors for visible points3D 
    p3d_feats = extract_points3d_descriptors(visible_p3d, feats_3d_path)
    if not p3d_feats:
        return None

    # reproject points3d to get GT
    img_depth_path = depth_path / f"{Path(query).stem}.h5"
    if not img_depth_path.exists():
        logger.warning(f"Depth map missing for {query}: {img_depth_path}")
        return None
    depth_map = load_depth(img_depth_path)
    matches0, matches1 = compute_ground_truth_matches(
        query_feats, p3d_feats, camera, depth_map, 
        reproj_thresh=args.reproj_thresh, 
        depth_rel_thresh=args.depth_thresh
    )

    gt_data = {
        "keypoints0": query_feats["keypoints"],               # shape (N,2)
        "descriptors0": query_feats["descriptors"].T,         # shape (N,D)
        "keypoints1": p3d_feats["keypoints"],                 # shape (M,3)
        "descriptors1": p3d_feats["descriptors"],             # shape (M,D)
        "matches0": matches0,                                 # shape (N,), matched 3D point index or -1
        "matches1": matches1,                                 # shape (M,), matched 2D keypoint index or -1
    }
    
    return gt_data

def process_scene(scene_path: Path, args):
    scene_name = scene_path.name
    logger.info(f"Generating Ground Truth Pairs for {scene_name}...")

    query_path = args.query_dir / scene_name
    query_names_file = query_path / "query_image_names.txt"
    query_pose_file = query_path / "query_image_cameras.txt"
    feats_3d_path = args.feature_dir / scene_name / "points3D_feats_cache.h5"
    feats_2d_path = args.sfm_dir / scene_name / "feats-superpoint-n2048.h5"
    covisibility_path = args.feature_dir / scene_name / "covisibility_results.pkl"
    depth_path = args.depth_dir / scene_name
    output_file = args.feature_dir / scene_name / "ground_truth.pkl"
    if not all([query_names_file.exists(), feats_3d_path.exists(), covisibility_path.exists()]):
        logger.error(f"Missing required input files for {scene_name}. Skipping.")
        return

    # load query names to a list
    with open(query_names_file, 'r') as f:
        query_list = [line.strip() for line in f]

    # load covisibility result, where covisibility_results[query_image] = {'unique_images': set of img_ids,
    # 'unique_points': np.array of point3D ids, 'max_distance': float}
    with open(covisibility_path, "rb") as f:
        covisibility_dict = pickle.load(f)

    # load query pose infos
    query_cams = load_query_cams(query_pose_file)
    scene_gt_data = {}

    match_counts = []

    # Process all queries in the scene
    for query in tqdm(query_list, desc=f"Processing Queries in {scene_name}"):
        gt_data = generate_gt_for_query(
            query, feats_2d_path, feats_3d_path, query_cams, covisibility_dict, depth_path, args
        )
        if gt_data is not None:
            scene_gt_data[query] = gt_data
            match_counts.append(np.sum(gt_data["matches0"] != -1))

    # Save to disk
    with open(output_file, "wb") as f:
        pickle.dump(scene_gt_data, f)

    # Print Summary
    if match_counts:
        logger.info(f"[{scene_name}] Ground Truth Generated for {len(scene_gt_data)} queries.")
        logger.info(f"[{scene_name}] Avg valid matches per query: {np.mean(match_counts):.2f}")
    logger.info(f"Saved to {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate 2D-3D Ground Truth Matches")
    parser.add_argument('--depth_dir', type=Path, required=True, help="Path to depth_undistorted")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query_sets")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to sfm outputs")
    parser.add_argument('--feature_dir', type=Path, required=True, help="Path to feature_results (covisibility & 3D feats)")
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--scene_list', type=Path, default=None)
    parser.add_argument('--reproj_thresh', type=float, default=3.0, help="Pixel distance threshold for 2D reprojection")
    parser.add_argument('--depth_thresh', type=float, default=0.1, help="Relative depth error threshold (10%)")
    args = parser.parse_args()

    scenes = []
    if args.scene_list:
        if not args.scene_list.exists():
            raise FileNotFoundError(f"Scene list file not found: {args.scene_list}")
        logger.info(f"Reading scenes from {args.scene_list}...")
        with open(args.scene_list, 'r') as f:
            scene_names = [line.strip() for line in f if line.strip()]
        for name in scene_names:
            scenes.append(args.depth_dir / name)
    elif args.scene:
        scenes = [args.depth_dir / args.scene]
    else:
        scenes = sorted([p for p in args.depth_dir.iterdir() if p.is_dir()])

    logger.info(f"Found {len(scenes)} scenes to process.")

    for scene_path in scenes:
        try:
            process_scene(scene_path, args)
        except Exception as e:
            logger.error(f"Failed to process {scene_path.name}: {e}", exc_info=True)
            continue

if __name__ == "__main__":
    main()