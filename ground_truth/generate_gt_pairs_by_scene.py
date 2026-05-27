import h5py
from pathlib import Path
import pickle
import numpy as np
from utils.utils import qvec2rotmat
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
import torch
torch.set_num_threads(1)

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

def compute_ground_truth_matches(
        query_feats, p3d_feats, camera, depth_map=None, reproj_thresh=3.0, depth_rel_thresh=0.1
        ):
    """
    return:
        matches0: (N2D,)
        matches1: (N3D,)
    """

    kpts2d = query_feats["keypoints"]      # (N2D, 2)
    pts3d = p3d_feats["keypoints"]       # (N3D, 3)

    N2D = kpts2d.shape[0]
    N3D = pts3d.shape[0]

    matches0 = -np.ones(N2D, dtype=int)
    matches1 = -np.ones(N3D, dtype=int)

    if N3D == 0:
        return matches0, matches1

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
    valid = z > 0 # check depth > 0

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

    projected = np.stack([u.flatten(), v.flatten()], axis=1)

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

    return matches0, matches1

IGNORE_FEATURE = -2
UNMATCHED_FEATURE = -1

def compute_ground_truth_matches_soft(query_feats, p3d_feats, camera, depth_map=None, pos_reproj_thresh=3.0, neg_reproj_thresh=8.0, pos_depth_thresh =0.1, neg_depth_thresh=0.25):
        """
        Vectorized PyTorch implementation of GT matching with Soft Thresholds.
        """

        IGNORE_FEATURE = -2
        UNMATCHED_FEATURE = -1

        kpts2d = torch.from_numpy(query_feats["keypoints"]).float() # (N2D, 2)
        pts3d = torch.from_numpy(p3d_feats["keypoints"]).float()    # (N3D, 3)

        N2D, N3D = kpts2d.shape[0], pts3d.shape[0]

        # Initialize with -1
        matches0 = torch.full((N2D,), UNMATCHED_FEATURE, dtype=torch.long)
        matches1 = torch.full((N3D,), UNMATCHED_FEATURE, dtype=torch.long)

        if N3D == 0:
            return matches0.numpy(), matches1.numpy()

        # Pose and intrinsics
        R = torch.from_numpy(qvec2rotmat(camera["qvec"])).float()
        t = torch.tensor(camera["tvec"]).float().view(3, 1)

        fx, fy, cx, cy = camera["intrinsics"]["params"][:4]
        width, height = camera["intrinsics"]["width"], camera["intrinsics"]["height"]

        # Vectorized 3D projection
        X = pts3d.T # (3, N3D)
        X_cam = R @ X + t # (3, N3D)
        
        z = X_cam[2, :]
        valid_z = z > 0
        
        # Avoid division by zero for invalid z
        z_safe = torch.where(valid_z, z, torch.ones_like(z))
        
        u = fx * (X_cam[0, :] / z_safe) + cx
        v = fy * (X_cam[1, :] / z_safe) + cy

        valid_proj = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)

        # Depth check arrays
        has_valid_depth = torch.zeros(N3D, dtype=torch.bool)
        rel_error = torch.full((N3D,), float('inf'))

        if depth_map is not None:
            # Only sample depth for points that landed inside the image bounds
            u_np, v_np = u[valid_proj].numpy(), v[valid_proj].numpy()
            
            if len(u_np) > 0:
                depth_real = sample_depth_bilinear(depth_map, u_np, v_np)
                depth_real_t = torch.from_numpy(depth_real).float()
                
                valid_d = depth_real_t > 0
                
                z_valid = z[valid_proj]
                rel_err_valid = torch.full_like(depth_real_t, float('inf'))
                rel_err_valid[valid_d] = torch.abs(z_valid[valid_d] - depth_real_t[valid_d]) / depth_real_t[valid_d]
                
                # Scatter back to the original N3D sized arrays
                rel_error[valid_proj] = rel_err_valid
                has_valid_depth[valid_proj] = valid_d
                # Filter the points that have depth but bigger than neg_thresh
                depth_is_totally_wrong = has_valid_depth & (rel_error > neg_depth_thresh)
                valid_proj &= (~depth_is_totally_wrong)
        # Combine u,v into (N3D, 2)
        projected = torch.stack([u, v], dim=1)
        
        # Vectorized Distance Matrix Calculation
        dist_matrix = torch.cdist(projected.unsqueeze(0), kpts2d.unsqueeze(0)).squeeze(0) # -> (N3D, N2D)
        
        # Mask out points that projected behind the camera or off-screen
        dist_matrix[~valid_proj] = float('inf')

        # Prepare for MNN assignment
        min_dist_3d_indices_for_2d = torch.argmin(dist_matrix, dim=0)
        min_dist_2d_indices_for_3d = torch.argmin(dist_matrix, dim=1)
        has_dist_mask = dist_matrix <= neg_reproj_thresh
        valid_2d_indices = torch.where(has_dist_mask)[1].unique()
        
        for idx2d in valid_2d_indices: # loop over valid 2d kpts
            # first assign all the existing mathes < neg_reproj_thresh as IGNORED
            has_dist_3d_idx = torch.where(has_dist_mask[:, idx2d])[0]
            if matches0[idx2d] == UNMATCHED_FEATURE:
                matches0[idx2d] = IGNORE_FEATURE
            for id in has_dist_3d_idx:
                if matches1[id] == UNMATCHED_FEATURE:
                    matches1[id] = IGNORE_FEATURE

            min_dist_3d_idx = min_dist_3d_indices_for_2d[idx2d]
            is_mutual = (min_dist_2d_indices_for_3d[min_dist_3d_idx] == idx2d)
            if is_mutual:
            # If mutual nearest neighbours, check if assigned as STRICT
                cur_min_dist = dist_matrix[min_dist_3d_idx, idx2d]
                r_err = rel_error[min_dist_3d_idx]
                valid_d = has_valid_depth[min_dist_3d_idx]
                if valid_d: # if has depth
                    if cur_min_dist <= pos_reproj_thresh and r_err <= pos_depth_thresh:
                        if (matches0[idx2d] in [UNMATCHED_FEATURE, IGNORE_FEATURE]) and \
                            (matches1[min_dist_3d_idx] in [UNMATCHED_FEATURE, IGNORE_FEATURE]):
                            matches0[idx2d] = min_dist_3d_idx
                            matches1[min_dist_3d_idx] = idx2d
                else: # if no depth provided
                    if matches1[min_dist_3d_idx] == UNMATCHED_FEATURE:
                        matches1[min_dist_3d_idx] = IGNORE_FEATURE

        return matches0.numpy(), matches1.numpy()

def load_depth(depth_path):
    with h5py.File(depth_path, 'r') as f:
        depth = f['depth'][:]
    return depth

def generate_gt_for_query(query_list, feats_2d_path, feats_3d_path, query_cams, covisibility_dict, depth_path):
    # extract SP keypoints descriptors of all the queries in one scene
    all_query_feats = {}
    query_set = set(query_list)
    with h5py.File(feats_2d_path, "r") as f_h5:
        all_keys = set(f_h5.keys())
        for img_name in query_set:
            if img_name in all_keys:
                ds = f_h5[img_name]
                all_query_feats[img_name] = {
                    "descriptors": ds["descriptors"][:],
                    "scores": ds["scores"][:],
                    "keypoints": ds["keypoints"][:]
                }

    # load 3d descriptors for all the 3d points
    points3d_feats = {}
    with h5py.File(feats_3d_path, "r") as f_h5:
        all_keys = set(f_h5.keys())
        for id in list(all_keys):
            ds = f_h5[str(id)]
            points3d_feats[str(id)] = {
                "descriptors": ds["descriptors"][:].reshape(1, 256),
                "keypoints": ds["keypoints"][:].reshape(1, 3),
                "scores": ds["scores"][:]
            }

    gt_data = {}
    for query in query_list:

        if query not in all_query_feats: 
            print(f"WARNING: {query} not found in all-query-feature list.")
            continue

        # query descriptors
        query_feats = all_query_feats[query]

        # load pose and camera of the query
        camera = query_cams[query] # qvec, tvec...

        # extract covisibility results of the query
        visible_p3d = covisibility_dict[query]["unique_points"]

        valid_p3d = [str(p) for p in visible_p3d if str(p) in points3d_feats]
        if not valid_p3d: 
            print(f"WARNING: No valid visible 3d points found for {query}.")
            continue

        # load keypoints and descriptors for visible points3D 
        current_p3d_feats = {
            "descriptors": np.vstack([points3d_feats[p]["descriptors"] for p in valid_p3d]),
            "keypoints": np.vstack([points3d_feats[p]["keypoints"] for p in valid_p3d]),
            "scores": [points3d_feats[p]["scores"] for p in valid_p3d]
        }

        # reproject points3d to get GT
        depth_map = load_depth(depth_path / f"{Path(query).stem}.h5")
        matches0, matches1 = compute_ground_truth_matches(
            query_feats, current_p3d_feats, camera, depth_map, reproj_thresh=3.0, depth_rel_thresh=0.1
        )
        # TODO: Wash data here, only keep matches > 0
        
        gt_data[query] = {
                "keypoints0": query_feats["keypoints"], # shape (N,2))
                "descriptors0": query_feats["descriptors"].T, # to shape(N,D)
                "keypoints1": current_p3d_feats["keypoints"], # shape (M,3)
                "descriptors1": current_p3d_feats["descriptors"],# shape(M,D)
                "matches0": matches0, # shape(N,), matched 3D point index or -1
                "matches1": matches1, # shape(M,), matched 2D keypoint index or -1
        }
    
    return gt_data

def generate_gt_for_query_soft(query, feats_2d_path, feats_3d_path, query_cams, covisibility_dict, depth_path, args):
    
    # Extract SP keypoints descriptors of all the queries in one scene
    with h5py.File(feats_2d_path, "r") as f_h5:
        if query not in f_h5.keys():
            print(f"WARNING: {query} not found in 2D feature list.")
            return None
            
        ds = f_h5[query]
        query_feats = {
            "descriptors": ds["descriptors"][:],
            "scores": ds["scores"][:],
            "keypoints": ds["keypoints"][:]
        }

    # Extract covisibility results
    visible_p3d = covisibility_dict[query]["unique_points"]

    # Load 3d descriptors for the visible points
    points3d_feats = {}
    with h5py.File(feats_3d_path, "r") as f_h5:
        all_keys = set(f_h5.keys())
        valid_p3d = [str(p) for p in visible_p3d if str(p) in all_keys]
        
        if not valid_p3d: 
            print(f"WARNING: No valid visible 3d points found for {query}.")
            return None
            
        for p in valid_p3d:
            ds = f_h5[p]
            points3d_feats[p] = {
                "descriptors": ds["descriptors"][:].reshape(1, 256),
                "keypoints": ds["keypoints"][:].reshape(1, 3),
                "scores": ds["scores"][:]
            }

    # Format the 3D features for matching
    current_p3d_feats = {
        "descriptors": np.vstack([points3d_feats[p]["descriptors"] for p in valid_p3d]),
        "keypoints": np.vstack([points3d_feats[p]["keypoints"] for p in valid_p3d]),
        "scores": [points3d_feats[p]["scores"] for p in valid_p3d]
    }

    # Load pose and camera
    camera = query_cams[query]

    # Load depth map
    depth_file = depth_path / f"{Path(query).stem}.h5"
    depth_map = load_depth(depth_file) if depth_file.exists() else None

    # Compute ground truth matches soft with ignored labels
    matches0, matches1 = compute_ground_truth_matches_soft(
        query_feats=query_feats, 
        p3d_feats=current_p3d_feats, 
        camera=camera, 
        depth_map=depth_map, 
        pos_reproj_thresh=args.pos_reproj_thresh, 
        neg_reproj_thresh=args.neg_reproj_thresh, 
        pos_depth_thresh=args.pos_depth_thresh, 
        neg_depth_thresh=args.neg_depth_thresh
    )
    
    # Return the dictionary for this query
    return {
        "keypoints0": query_feats["keypoints"],       # shape (N,2)
        "descriptors0": query_feats["descriptors"].T, # to shape (N,D)
        "keypoints1": current_p3d_feats["keypoints"], # shape (M,3)
        "descriptors1": current_p3d_feats["descriptors"], # shape (M,D)
        "matches0": matches0, # shape (N,), matched 3D point index, -1, or -2
        "matches1": matches1, # shape (M,), matched 2D keypoint index, -1, or -2
    }