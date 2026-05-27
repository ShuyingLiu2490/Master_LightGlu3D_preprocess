import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm
from hloc.utils import read_write_model as rw
from collections import defaultdict
import os

def extract_3d_descriptors(points3D, images, h5_path):
    """
    Compute the averaged features for 3D points
    Returns: { p3d_id: {'descriptors': ..., 'keypoints': ..., 'scores': ...} }
    """
    # First load image <-> 3D point correspondences in to dict
    image_to_obs = defaultdict(list)

    for p3d_id, p3d_obj in points3D.items():
        for img_id, p2d_idx in zip(p3d_obj.image_ids, p3d_obj.point2D_idxs):
            if img_id in images:
                image_to_obs[img_id].append((p3d_id, p2d_idx))

    p3d_feature_dict = defaultdict(lambda: {
        "descriptors": [],
        "scores": [],
        "xyz": None
    })
    # Loop over image ids and inner loop for current image cooresponding 3Ds
    with h5py.File(h5_path, "r") as f_h5:
        for img_id, observations in tqdm(image_to_obs.items()):
            
            img_name = images[img_id].name
            if img_name not in f_h5:
                print(f"WARNING: {img_name} not found in {h5_path}")
                continue

            ds = f_h5[img_name]
            descriptors = ds["descriptors"][:]
            scores = ds["scores"][:]

            for p3d_id, p2d_idx in observations:
                p3d_feature_dict[p3d_id]["descriptors"].append(
                    descriptors[:, p2d_idx]
                )
                p3d_feature_dict[p3d_id]["scores"].append(
                    scores[p2d_idx]
                )
                p3d_feature_dict[p3d_id]["xyz"] = points3D[p3d_id].xyz

    final_dict = {}

    for p3d_id, data in p3d_feature_dict.items():
        descs = np.stack(data["descriptors"], axis=0)
        avg_desc = descs.mean(axis=0)
        avg_desc /= (np.linalg.norm(avg_desc) + 1e-6)

        avg_score = np.mean(data["scores"])

        final_dict[str(p3d_id)] = {
            "descriptors": avg_desc.reshape(1, -1), # shape(num,256)
            "keypoints": data["xyz"].reshape(1, 3),
            "scores": np.array([avg_score])
        }
    print("Finished 3D features comptation.")

    return final_dict


def save_3d_features_to_h5(feature_dict, output_path):
    parent_dir = os.path.dirname(output_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        
    with h5py.File(output_path, "w") as f:
        for p3d_id, data in feature_dict.items():
            grp = f.create_group(p3d_id)
            grp.create_dataset('descriptors', data=data['descriptors'])
            grp.create_dataset('keypoints', data=data['keypoints'])
            grp.create_dataset('scores', data=data['scores'])

if __name__ == "__main__":

    root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
    scene_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    for scene in scene_names[:12]:

        print(f"Start averaged feature computation for scene {scene}...")
        sfm_dir = Path("/proj/vlarsson/outputs/sfm") / scene / "sfm_superpoint+lightglue"
        output_dir = Path("/proj/vlarsson/outputs/midterm_results/") / scene

        _, images, points3D = rw.read_model(sfm_dir, ext=".bin")
        h5_path = sfm_dir.parent / "feats-superpoint-n2048.h5"
        p3d_feats = extract_3d_descriptors(points3D, images, h5_path)
        print(f"Extracted features for {len(p3d_feats)} 3D points.")
        # print one example
        # keys = [key for key in p3d_feats.keys()]
        # print(p3d_feats[keys[0]])
        cached_feats_path = output_dir / "points3D_feats_cache.h5"
        save_3d_features_to_h5(p3d_feats, cached_feats_path)
        print(f"Averaged feature for scene {scene} saved to {cached_feats_path}.")