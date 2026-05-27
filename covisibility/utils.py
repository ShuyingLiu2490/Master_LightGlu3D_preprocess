import numpy as np

def map_img_to_points3d(image_name: str, images_dict) -> np.ndarray:
    '''
    Maps an image name to its corresponding 3D point IDs in the SfM model.
    '''
    for img in images_dict.values():
        if img.name == image_name:
            return img.point3D_ids[img.point3D_ids != -1]
    raise ValueError(f"{image_name} not found.")


def map_img_name_to_id(image_name: str, images_dict) -> int:
    '''
    Maps an image name to its corresponding image ID in the SfM model.
    '''
    for img_id, img in images_dict.items():
        if img.name == image_name:
            return img_id
    raise ValueError(f"{image_name} not found.")    

def qvec2rotmat(qvec):
    """Convert quaternion vector to rotation matrix.
    Args:
        qvec: Quaternion vector (4,).
    Returns:
        Rotation matrix (3, 3).
    """
    w, x, y, z = qvec
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])
    return R
