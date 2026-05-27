from pathlib import Path
import torch
import numpy as np
import os
import warnings
import torch.multiprocessing as mp
warnings.filterwarnings("ignore", category=FutureWarning)

from hloc import extract_features, match_features

def extract_pairs_to_list(query_file_path, scene_info_npz_path, overlap_thres=[0.3, 0.9]):

    print(f"Using overlap threshold range: {overlap_thres}")

    full_data = np.load(scene_info_npz_path, allow_pickle=True)
    overlap_matrix = full_data['overlap_matrix']
    image_paths = full_data['image_paths']

    with open(query_file_path, 'r') as f:
        query_image_names = [line.strip() for line in f.readlines()]

    valid_pairs = np.argwhere((overlap_matrix >= overlap_thres[0]) & (overlap_matrix < overlap_thres[1]))
    valid_pairs_name = []
    for pair in valid_pairs:
        name1 = os.path.basename(image_paths[pair[0]])
        name2 = os.path.basename(image_paths[pair[1]])
        if name1 in query_image_names or name2 in query_image_names:
            continue

        valid_pairs_name.append((name1, name2))

    return valid_pairs_name


root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
outputs = Path("/proj/vlarsson/outputs/sfm/")
html_save_dir =Path("/home/x_jiagu/degree_project/SfM_htmls")

overlap_thres = [0.3, 0.95]  # define your overlap threshold range here

feature_conf = {
    'output': 'feats-superpoint-n2048',
    'model': {
        'name': 'superpoint',
        'nms_radius': 4,
        'max_keypoints': 2048, 
    },
    'preprocessing': {
        'grayscale': True,
        'resize_max': 1600, 
        "resize_force": True,
    },
}

matcher_conf = match_features.confs["superpoint+lightglue"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scene_names = sorted([
    p.name
    for p in root.iterdir()
    if p.is_dir()
])

def process_scene(scene): # change the slice to process more scenes
    print(f"Start processing scene: {scene}...")

    images_path = root / scene / "images"

    output_dir = outputs / scene
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using GPU for processing feature extraction and matching.")
    sfm_pairs = output_dir / "pairs-covisibility.txt"

    # Step 0: Get image pairs from dataset info: skip all queries
    pair_load = extract_pairs_to_list(outputs.parent / 'query_sets' / scene / 'query_image_names.txt',
                                           root.parent / 'scene_info' / f'{scene}.npz', 
                                           overlap_thres)

    with sfm_pairs.open("w") as f:
        for im1, im2 in pair_load:
            f.write(f"{im1} {im2}\n")

    print(f"Finished similar pairs retrival. Processed {len(pair_load)} pairs.")

    # Step 1: Feature extraction
    feature_path = extract_features.main(feature_conf, images_path, output_dir)

    # Step 2: Pairwise matching
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], output_dir)

    print(f"Secene {scene} feature extraction and matching on GPU completed.")

def worker(scene, semaphore):
    with semaphore:
        process_scene(scene)


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    semaphore = mp.Semaphore(4)
    processes = []

    for scene in scene_names[:12]:
        p = mp.Process(target=worker, args=(scene, semaphore))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

