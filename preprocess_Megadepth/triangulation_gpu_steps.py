import argparse
import logging
import os
import warnings
import numpy as np
from pathlib import Path
import torch.multiprocessing as mp
from hloc import extract_features, match_features
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurations
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

def extract_pairs_to_list(query_file_path: Path, scene_info_npz_path: Path, overlap_thres: list) -> list:
    
    full_data = np.load(scene_info_npz_path, allow_pickle=True)
    overlap_matrix = full_data['overlap_matrix']
    image_paths = full_data['image_paths']

    with open(query_file_path, 'r') as f:
        query_image_names = {line.strip() for line in f.readlines()}

    valid_pairs = np.argwhere((overlap_matrix >= overlap_thres[0]) & (overlap_matrix < overlap_thres[1]))
    valid_pairs_name = []
    for pair in valid_pairs:
        name1 = os.path.basename(image_paths[pair[0]])
        name2 = os.path.basename(image_paths[pair[1]])
        # Skip if either image is a query image
        if name1 in query_image_names or name2 in query_image_names:
            continue
            
        valid_pairs_name.append((name1, name2))

    return valid_pairs_name

def process_scene(scene_path: Path, args):
    scene_name = scene_path.name
    logger.info(f"Start processing scene (GPU): {scene_name}...")

    images_path = scene_path / "images"
    output_dir = args.outputs / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    sfm_pairs = output_dir / "pairs-covisibility.txt"
    query_file_path = args.query_dir / scene_name / 'query_image_names.txt'
    scene_info_path = args.dataset.parent / 'scene_info' / f'{scene_name}.npz'

    # Step 0: Get image pairs from dataset info: skip all queries
    logger.info(f"[{scene_name}] Generating pairs using overlap threshold {args.overlap_thres}")
    pair_load = extract_pairs_to_list(query_file_path, scene_info_path, args.overlap_thres)

    with sfm_pairs.open("w") as f:
        for im1, im2 in pair_load:
            f.write(f"{im1} {im2}\n")
    logger.info(f"[{scene_name}] Processed {len(pair_load)} valid reference pairs.")

    # Step 1: Feature extraction
    logger.info(f"[{scene_name}] Extracting SuperPoint features...")
    extract_features.main(feature_conf, images_path, output_dir)

    # Step 2: Pairwise matching
    logger.info(f"[{scene_name}] Matching features with LightGlue...")
    match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], output_dir)

    logger.info(f"Scene {scene_name} feature extraction and matching on GPU completed.")

def worker(scene_path: Path, args, semaphore):
    with semaphore:
        try:
            process_scene(scene_path, args)
        except Exception as e:
            logger.error(f"Failed to process {scene_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="GPU Feature Extraction and Matching (Multiprocessing)")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Undistorted_SfM")
    parser.add_argument('--outputs', type=Path, required=True, help="Path to output sfm directory")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query sets directory")
    parser.add_argument('--scene', type=str, default=None, help="Process a single scene ID, eg: 0000, 0001, etc.")
    parser.add_argument('--scene_list', type=Path, default=None)
    parser.add_argument('--min_overlap', type=float, default=0.3)
    parser.add_argument('--max_overlap', type=float, default=0.95)
    parser.add_argument('--num_workers', type=int, default=4, help="Number of concurrent GPU processes")
    args = parser.parse_args()
    args.overlap_thres = [args.min_overlap, args.max_overlap]

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset}")

    # Determine scenes to process
    scenes = []
    if args.scene_list:
        if not args.scene_list.exists():
            raise FileNotFoundError(f"Scene list file not found: {args.scene_list}")
        logger.info(f"Reading scenes from {args.scene_list}...")
        with open(args.scene_list, 'r') as f:
            scene_names = [line.strip() for line in f if line.strip()]
        for name in scene_names:
            scenes.append(args.dataset / name)
    elif args.scene:
        scenes = [args.dataset / args.scene]
    else:
        scenes = sorted([p for p in args.dataset.iterdir() if p.is_dir() and (p / "images").exists()])

    logger.info(f"Found {len(scenes)} scenes to process.")
    logger.info(f"Using {args.num_workers} concurrent workers.")

    # Initialize multiprocessing
    mp.set_start_method("spawn", force=True)
    semaphore = mp.Semaphore(args.num_workers)
    processes = []

    # Launch workers
    for scene_path in scenes:
        p = mp.Process(target=worker, args=(scene_path, args, semaphore))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    logger.info("All GPU processing completed.")


if __name__ == "__main__":
    main()