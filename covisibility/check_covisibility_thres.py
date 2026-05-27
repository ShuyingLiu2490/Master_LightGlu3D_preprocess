import pycolmap
from pathlib import Path
import numpy as np
import os

def extract_pairs_to_list(npz_path, overlap_thres=[0.1, 0.5]):

    full_data = np.load(npz_path, allow_pickle=True)
    overlap_matrix = full_data['overlap_matrix']
    valid_pairs = np.argwhere((overlap_matrix >= overlap_thres[0]) & (overlap_matrix < overlap_thres[1]))
    valid_pairs_path = full_data['image_paths'][valid_pairs]
    valid_pairs_name = [(os.path.basename(pair[0]), os.path.basename(pair[1])) for pair in valid_pairs_path]

    return valid_pairs_name

# colmap model

def count_pairs(scene, threshold, root, ref_path):
    model = pycolmap.Reconstruction(ref_path)
    model_image_names = {img.name for _, img in model.images.items()}

    # covisibity pairs
    pair_load = extract_pairs_to_list(root.parent / 'scene_info' / f'{scene}.npz', threshold)
    # sfm_pairs_path = Path(f"./outputs/sfm/{scene}/pairs-covisibility.txt")
    pair_image_names = set()
    pair_image_names.update([name for pair in pair_load for name in pair])
    # with open(sfm_pairs_path, "r") as f:
    #     for line in f:
    #         pair_image_names.update(line.strip().split())

    # check missing images
    missing_in_model = pair_image_names - model_image_names
    print(f"Infos for Scene: {scene}")
    print(f"Covisibility threshold: {threshold}")
    print(f"{len(pair_load)} image pairs found")
    print(f"{len(pair_image_names)} images found in Image Pairs")
    print(f"{len(model_image_names)} images found in Reference Model")
    print(f"Number of images not exist in Model: {len(missing_in_model)}")

    if missing_in_model:
        print("The missing image name:", list(missing_in_model)[:5])

    return 


root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
scene_names = sorted([
    p.name
    for p in root.iterdir()
    if p.is_dir()
])

thresholds = [[0.3, 0.95]]

for scene in scene_names[:5]:
    ref_path = Path(f"/proj/vlarsson/datasets/megadepth/Undistorted_SfM/{scene}/sparse")
    for thres in thresholds:
        print("---------------------------------------------------")
        count_pairs(scene, thres, root, ref_path)
        print("---------------------------------------------------")
        