import h5py
from pathlib import Path
import pickle
import numpy as np
from utils import qvec2rotmat
import torch
import torch.nn.functional as F

def extract_query_decriptors(img_name, h5_path):

    query_feature_dict = {}
    with h5py.File(h5_path, "r") as f_h5:

        if img_name not in f_h5:
            print(f"WARNING: {img_name} not found in {h5_path}")
            return query_feature_dict
        ds = f_h5[img_name]

        query_feature_dict["descriptors"] = ds["descriptors"][:]
        query_feature_dict["scores"] = ds["scores"][:]
        query_feature_dict["keypoints"] = ds["keypoints"][:]

    print(f"Collected descriptors for query image {img_name}.")
    
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
    descriptors =[]
    keypoints = []
    scores = []
    with h5py.File(h5_path, "r") as f_h5:
        for id in points3d:
            id = str(id)
            if id not in f_h5:
                print(f"WARNING: {id} not found in {h5_path}")
                continue
            ds = f_h5[id]
            descriptors.append(ds["descriptors"][:].reshape(1,256))
            keypoints.append(ds["keypoints"][:].reshape(1,3))
            scores.append(ds["scores"][:])

    point3d_feature_dict["descriptors"] = np.vstack(descriptors)
    point3d_feature_dict["scores"] = scores
    point3d_feature_dict["keypoints"] = np.vstack(keypoints)

    print(f"Collected descriptors for {np.shape(point3d_feature_dict['keypoints'])[0]} 3D points.")

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

    projected = np.stack([u, v], axis=1)

    # find the nearest 2D keypoint
    for idx3d, proj_pt in zip(valid_indices, projected):

        dists = np.linalg.norm(kpts2d - proj_pt, axis=1)
        min_idx = np.argmin(dists)

        if dists[min_idx] < reproj_thresh:

            if matches0[min_idx] == -1: # in case to rewrite, only register the first matched pair
                matches0[min_idx] = idx3d
                matches1[idx3d] = min_idx

    return matches0, matches1

def load_depth(depth_path):
    with h5py.File(depth_path, 'r') as f:
        depth = f['depth'][:]
    return depth

def generate_gt_for_query(query, feats_2d_path, feats_3d_path, query_cams, covisibility_dict, depth_path):
    # extract SP keypoints descriptors of the query
    query_feats = extract_query_decriptors(query, feats_2d_path)

    # load pose and camera of the query
    camera = query_cams[query] # qvec, tvec...

    # extract covisibility results of the query
    visible_p3d = covisibility_dict[query]["unique_points"]

    # load keypoints and descriptors for visible points3D 
    p3d_feats = extract_points3d_descriptors( visible_p3d, feats_3d_path)

    # reproject points3d to get GT
    depth_map = load_depth(depth_path / f"{Path(query).stem}.h5")
    matches0, matches1 = compute_ground_truth_matches(
        query_feats, p3d_feats, camera, depth_map, reproj_thresh=3.0, depth_rel_thresh=0.1
    )
    gt_data = {
            "keypoints0": query_feats["keypoints"], # shape (N,2))
            "descriptors0": query_feats["descriptors"].T, # to shape(N,D)
            "keypoints1": p3d_feats["keypoints"], # shape (M,3)
            "descriptors1": p3d_feats["descriptors"],# shape(M,D)
            "matches0": matches0, # shape(N,), matched 3D point index or -1
            "matches1": matches1, # shape(M,), matched 2D keypoint index or -1
    }
    
    return gt_data

if __name__ == "__main__":
    output_dir = Path("/proj/vlarsson/outputs") 
    scene = "0000"
    query_path = output_dir / "query_sets" / scene
    query_names = query_path / "query_image_names.txt"
    query_pose = query_path / "query_image_cameras.txt"

    feats_3d_path = output_dir / "midterm_results" / scene / "points3D_feats_cache.h5" # averaged descriptors for all 3D points
    feats_2d_path = output_dir / "sfm" / scene / "feats-superpoint-n2048.h5" # cached SP descriptors
    covisibility_result_path = output_dir / "midterm_results" / scene / "covisibility_results.pkl" # covisibility results for all queries
    depth_path = Path("/proj/vlarsson/datasets/megadepth/depth_undistorted") / scene # depth maps for all queries
    # load query names to a list
    with open(query_names, 'r') as f:
        query_list = [line.strip() for line in f]

    # load covisibility result, where covisibility_results[query_image] = {'unique_images': set of img_ids,
    # 'unique_points': np.array of point3D ids, 'max_distance': float}
    with open(covisibility_result_path, "rb") as f:
        covisibility_dict = pickle.load(f)

    # load query pose infos
    query_cams = load_query_cams(query_pose)
    gt_data = {}
    for query in query_list[:1]:
        gt_data[query] = generate_gt_for_query(
            query, feats_2d_path, feats_3d_path, query_cams, covisibility_dict, depth_path
            )

    # print(f"Shape of keypoints0: {np.shape(gt_data['keypoints0'])}")
    # print(f"Shape of descriptors0: {np.shape(gt_data['descriptors0'])}")
    # print(f"Shape of keypoints1: {np.shape(gt_data['keypoints1'])}")
    # print(f"Shape of descriptors1: {np.shape(gt_data['descriptors1'])}")
    # print(f"Shpae of matches0: {np.shape(gt_data['matches0'])}")
    # print(f"Shpae of matches1: {np.shape(gt_data['matches1'])}")
    # print(gt_data)
    # print(matches0)
    # print("num matches0:", np.sum(matches0 != -1))
    # print(matches1)
    # print("num matches1:", np.sum(matches1 != -1))
    


