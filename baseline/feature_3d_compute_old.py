import torch
import numpy as np
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint, DISK 
from lightglue.utils import load_image, rbd

def load_3d_2d_tracks(points3D_txt_path):
    """
    Return a numpy array of 2D tracks with shape(num_points3D, 1, variable_length), 
    each track is:
    [(IMAGE_ID, POINT2D_IDX), ...]
    """
    matches_3d_2d = []

    with open(points3D_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue

            parts = line.strip().split()
            track_data = parts[8:]

            assert len(track_data) % 2 == 0

            track = []
            for i in range(0, len(track_data), 2):
                image_id = int(track_data[i])
                point2d_idx = int(track_data[i + 1])
                track.append((image_id, point2d_idx))

            matches_3d_2d.append(track)

    return matches_3d_2d # shape (num_points3D, 1, variable_length)


def load_image_points2d(image_txt_path):
    """
    Returns:
      image_points2d = {
        IMAGE_ID: [(x, y), ...]
      }
    """
    image_points2d = {}

    with open(image_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue

            parts = line.strip().split()
            if len(parts) == 10:
                image_id = int(parts[0])
                continue  # skip imageID lines

            # second line contains POINTS2D

            points = []
            for j in range(0, len(parts), 3):
                x = float(parts[j])
                y = float(parts[j + 1])
                points.append((x, y))  # ignore POINT3D_ID

            image_points2d[image_id] = points

    return image_points2d


# temp: positional encoding: simply remove one axis, default z-axis
def pos_encode(point_3d : np.array, scaler=1, exclude_axis='z', showplot=False):
    if exclude_axis=='z':
        points_2d = point_3d[:, :2] # take x, y corrdinates
    elif exclude_axis=='y':
        points_2d = np.stack((point_3d[:, 0], point_3d[:, -1]), axis=1) # take x, z corrdinates
    elif exclude_axis=='x':
        points_2d = point_3d[:, 1:] # take y, z corrdinates
    else:
        raise ValueError("exclude_axis must be 'x', 'y', or 'z'")
    x_min, y_min = np.min(points_2d, axis=0)
    x_max, y_max = np.max(points_2d, axis=0)
    image_width =  max(int((x_max - x_min) * scaler), 1)
    image_height =  max(int((y_max - y_min) * scaler), 1)

    def map_to_pixel(coords, min_val, max_val, size):
        """map every keypoints_2d(float) into a pixel"""
        if max_val - min_val < 1e-10: 
            return np.full(len(coords), size // 2)
        normalized = (coords - min_val) / (max_val - min_val)
        pixels = (normalized * (size - 1)).astype(int)
        return np.clip(pixels, 0, size - 1)
    
    pixel_x = map_to_pixel(points_2d[:, 0], x_min, x_max, image_width)
    pixel_y = map_to_pixel(points_2d[:, 1], y_min, y_max, image_height)
    keypoints_2d = np.column_stack((pixel_x, pixel_y))
    if showplot==True:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.scatter(keypoints_2d[:, 0], keypoints_2d[:, 1], s=5, alpha=0.7, color='blue')
        # plt.tight_layout()
        plt.axis('equal')
        plt.show()
    return keypoints_2d, image_width, image_height


def cache_features(image_txt_path, image_ref_dir, extractor):
    '''
    Returns:
      image_cache = {
        IMAGE_ID: {
            'keypoints': np.array of shape (N, 2),
            'descriptors': np.array of shape (N, D),
            'scores': np.array of shape (N,),
            'size' : np.array of shape (2,) (height, width)
        }
        ...
      }

    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
    image_cache = {}
    image_id_to_name = {}

    with open(image_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue

            parts = line.strip().split()
            if len(parts) > 10:
                continue  # skip POINTS2D lines
            image_id_to_name[int(parts[0])] = parts[9]

    print(f"{len(image_id_to_name)} images to process in total")

    # print("Pre-loading image features...")
    for img_id in tqdm(image_id_to_name.keys()):
        img_path = image_ref_dir / f"{image_id_to_name[img_id]}" 
        if not img_path.exists():
            print(f"Error: Image {img_id} does not exist")
            continue
        try:
            image = load_image(img_path).to(device)
            with torch.no_grad():
                feats = extractor.extract(image)
            image_cache[img_id] = {
                'keypoints': feats['keypoints'].cpu().numpy(),                
                'descriptors': feats['descriptors'].cpu().numpy(),
                'scores': feats['keypoint_scores'].cpu().numpy(),
                'size' : feats['image_size'].cpu().numpy()
            }
        except Exception as e:
            print(f"Error occured when processing image {img_id}: {e}")
            continue

    return image_cache


def compute_averaged_features(
        points3D_txt_path, image_txt_path, image_cache, points3D, 
        filter_3d_mask, descriptor_dim=128, th=2.0
        ):
        # descriptor_dim 256 for superpoint, 128 for disk
    '''
    Returns:
        point3d_features = {
            POINT3D_ID: {
                'descriptor': np.array of shape (D,),
                'keypoint_scores' : float,
                'num_observations': int,
                'point_3d': np.array of shape (3,)
            }
            ...
        }    
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print("Start loading reference image features...")
    tracks = load_3d_2d_tracks(points3D_txt_path)

    matches_3d_2d = [
        t for t, keep in zip(tracks, filter_3d_mask) if keep
    ]
    # matches_3d_2d = load_3d_2d_tracks(points3D_txt_path)[filter_3d_mask]
    image_points2d = load_image_points2d(image_txt_path)
    # image_cache = cache_features(image_txt_path, image_ref_dir, extractor)
    print("Computing averaged features for 3D points...")
    point3d_features = {}
    descriptor = []
    score = []
    num_observations = []
    for point_data in tqdm(matches_3d_2d):
        # point_data = matches_3d_2d[i]
        descriptors_list, scores_list = [], []
        for img_id, point2D_idx in point_data:
            if img_id not in image_cache:
                # print(f"Warning: Image {img_id} not found in cache, skipping.")
                continue
                    
            feats = image_cache[img_id]
            keypoints = feats['keypoints']
            descriptors = feats['descriptors']
            scores = feats['scores']
            if len(keypoints) == 0:
                continue

            if keypoints.ndim == 3 and keypoints.shape[0] == 1:
                keypoints = keypoints[0]
            if descriptors.ndim == 3 and descriptors.shape[0] == 1:
                descriptors = descriptors[0]
            if scores.ndim == 2 and scores.shape[0] == 1:
                scores = scores[0]
            
            actual_coord = image_points2d[img_id][point2D_idx]
            distances = np.linalg.norm(keypoints - actual_coord, axis=1)
            closest_idx = np.argmin(distances)
        
            if np.min(distances) < th and closest_idx < len(distances):  # Pixels, adjustable threshold
                descriptors_list.append(descriptors[closest_idx])
                scores_list.append(scores[closest_idx])
                    
        if descriptors_list:
            avg_descriptor = np.mean(descriptors_list, axis=0)
            avg_score = np.mean(scores_list, axis=0)
            descriptor.append(avg_descriptor.flatten())
            score.append(avg_score)
            num_observations.append(len(descriptors_list))
            # point3d_features[i] = {
            #     'descriptor': avg_descriptor,
            #     'keypoint_scores' : avg_score,
            #     'num_observations': len(descriptors_list),
            #     'point_3d': points3D[i, :] 
            # }
        else:
            descriptor.append(np.zeros(descriptor_dim))
            score.append(0.0)
            num_observations.append(0)
            # point3d_features[i] = {
            #     'descriptor': np.zeros(descriptor_dim),
            #     'keypoint_scores' : 0,
            #     'num_observations': 0,
            #     'point_3d': points3D[i, :]
            # }
    
    ref_real = [ k for k in num_observations if k!= 0]
    print(f"{len(ref_real)} of {len(num_observations)} 3D points get references")

    descriptor = np.array(descriptor)    # (N, descriptor_dim)
    score = np.array(score)              # (N,)
    
    # TODO: temp setting, to be removed. Convert 3D keypoints into an artificial image
    keypoints_2d, image_width, image_height = pos_encode(points3D, scaler=3, exclude_axis='z', showplot=True)
    
    # convert to torch.tensor format
    keypoints_tensor = torch.from_numpy(keypoints_2d).float().to(device).unsqueeze(0)      # (1, N, 2)
    descriptors_tensor = torch.from_numpy(descriptor).float().to(device).unsqueeze(0)     # (1, N, dim)
    scores_tensor = torch.from_numpy(score).float().to(device).unsqueeze(0)              # (1, N)
    image_size_tensor = torch.tensor([image_width, image_height], dtype=torch.float).to(device).unsqueeze(0)  # (1, 2)
    
    point3d_features = {
        'keypoints': keypoints_tensor,
        'descriptors': descriptors_tensor, 
        'keypoint_scores': scores_tensor,
        'image_size': image_size_tensor
    }
    
    print(f"Converted {len(num_observations)} 3D points to virtual image of size {image_width}x{image_height}.")

    return point3d_features

