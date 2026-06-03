import torch
import numpy as np

def compute_pr_baseline(matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, q_camera, device):
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