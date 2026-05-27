import numpy as np

def quaternion_to_rotation_matrix(q):
    """
    q: (qw, qx, qy, qz)  -- COLMAP order
    """
    qw, qx, qy, qz = q

    # normalization
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)]
    ])

    return R


def load_image_pose(images_txt_path, retrieval_image_name):
    """
    Return R, t for the best matched image
    """
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('#') or len(line.strip()) == 0:
            continue

        parts = line.strip().split()
        if len(parts) > 10:
            continue  # skip POINTS2D lines

        image_name = parts[9]
        if image_name == retrieval_image_name:
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])

            R = quaternion_to_rotation_matrix((qw, qx, qy, qz))
            t = np.array([tx, ty, tz]).reshape(3, 1)

            return R, t

    raise ValueError(f"Image {retrieval_image_name} not found in images.txt")


def load_points3D(points3D_txt_path):
    """
    Return Nx3 array of 3D points in world coordinates
    """
    points = []

    with open(points3D_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue

            parts = line.strip().split()
            X, Y, Z = map(float, parts[1:4])
            points.append([X, Y, Z])

    return np.array(points)  # shape: (N, 3)


def transform_points_to_camera(points_world, R, t):
    """
    points_world: (N, 3)
    """
    # Xc = R * Xw + t
    points_cam = (R @ points_world.T + t).T
    
    return points_cam


def quantile(points3D, quantile_value=0.95):
    """
    Compute the quantile of the 3D points.
    """
    upper_bound = np.quantile(points3D, quantile_value, axis=0)
    lower_bound = np.quantile(points3D, 1 - quantile_value, axis=0)
    filter_mask = (
        (points3D[:, 0] >= lower_bound[0]) & (points3D[:, 0] <= upper_bound[0]) &
        (points3D[:, 1] >= lower_bound[1]) & (points3D[:, 1] <= upper_bound[1]) &
        (points3D[:, 2] >= lower_bound[2]) & (points3D[:, 2] <= upper_bound[2]) &
        (points3D[:, 2] > 0)
    )
    # quantiled_points = points3D[filter_mask]
    return filter_mask


def sphere_normalize(points3D):
    """
    Normalize the 3D points to fit within a unit sphere.
    """
    centroid = np.quantile(points3D, 0.5, axis=0) # median or mean?
    # centroid = np.mean(points3D, axis=0)
    centered_points = points3D - centroid
    max_distance = np.max(np.linalg.norm(centered_points, axis=1))
    normalized_points = centered_points / max_distance * 1000 # scale to 1000 units

    return normalized_points


def project_pointcloud_to_query_view(
    images_txt_path,
    points3D_txt_path,
    matched_image_name
):
    # 1. load pose
    R, t = load_image_pose(images_txt_path, matched_image_name)

    # 2. load point cloud
    points_world = load_points3D(points3D_txt_path)

    # 3. transform
    points_camera = transform_points_to_camera(points_world, R, t)

    # 4. 3D points processing: filter points behind the camera; quantile-based normalization
    # in_front_indices = points_camera[:, 2] > 0
    # points_camera = points_camera[in_front_indices]
    mask = quantile(points_camera, quantile_value=0.975)
    points_camera = points_camera[mask]
    points_camera = sphere_normalize(points_camera)

    return {
        "camera_matrix": np.hstack([R, t]),
        "points_camera": points_camera,
        "filter_mask": mask
    }
