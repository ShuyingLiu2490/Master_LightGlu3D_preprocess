import argparse
import h5py
import logging
import os
import shutil
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from hloc import triangulation
from hloc.utils import read_write_model as rw
from utils.visual_sfm_3d import visualize_sfm_3d
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def filter_reference_model(reference_model: Path, output_model: Path, valid_images: set):
    cameras, images, points3D = rw.read_model(reference_model)
    assert reference_model != output_model

    images_filtered = {}

    for image_id, img in images.items():
        if img.name not in valid_images:
            continue
        
        new_point3D_ids = np.full_like(img.point3D_ids, -1)

        new_img = type(img)(
            id=img.id,
            qvec=img.qvec,
            tvec=img.tvec,
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=new_point3D_ids
        )
        images_filtered[image_id] = new_img

    points3D_filtered = {}

    rw.write_model(cameras, images_filtered, points3D_filtered, output_model, ext='.bin')


def process_scene(scene_path: Path, output_path: Path, html_save_dir: Path):
    scene_name = scene_path.name
    logger.info(f"Start processing scene (CPU): {scene_name}...")
    
    images_dir = scene_path / "images"
    scene_output_dir = output_path / scene_name
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    # Expected input files from the GPU step
    sfm_pairs = scene_output_dir / "pairs-covisibility.txt"
    feature_path = scene_output_dir / "feats-superpoint-n2048.h5"
    match_path = scene_output_dir / "feats-superpoint-n2048_matches-superpoint-lightglue_pairs-covisibility.h5"
    
    # Output directories
    sfm_dir = scene_output_dir / "sfm_superpoint+lightglue"
    filtered_model_path = scene_output_dir / "sparse_filtered"

    if not (sfm_pairs.exists() and feature_path.exists() and match_path.exists()):
        logger.warning(f"Skipping {scene_name}: Missing features/matches from GPU step.")
        return

    if (sfm_dir / "points3D.bin").exists():
        logger.info(f"Skipping {scene_name}: Triangulated output already exists at {sfm_dir}")
        return

    # Safely locate the sparse reference model
    reference_model = scene_path / "sparse"
    if not (reference_model / "cameras.bin").exists():
        if (scene_path / "sparse" / "0").exists():
            reference_model = scene_path / "sparse" / "0"
    if not reference_model.exists():
        logger.warning(f"Skipping {scene_name}: No sparse reference model found.")
        return

    # Step 1: Get valid images from matches
    logger.info(f"[{scene_name}] Extracting valid images from matches...")
    with h5py.File(match_path, 'r') as f:
        groups = list(f.keys())
        valid_images = set(groups)
        for group in groups:
            valid_images.update(f[group].keys())

    logger.info(f"[{scene_name}] Number of images with matches: {len(valid_images)}")

    # Step 2: Filter reference model, only keeping images with matches
    logger.info(f"[{scene_name}] Filtering reference model...")
    filtered_model_path.mkdir(exist_ok=True)
    filter_reference_model(reference_model, filtered_model_path, valid_images)

    # Step 3: Triangulation to obtain 3D model
    logger.info(f"[{scene_name}] Running hloc triangulation...")
    model = triangulation.main(sfm_dir, filtered_model_path, images_dir, sfm_pairs, feature_path, match_path)
    # Cleanup intermediate files
    shutil.rmtree(filtered_model_path)
    if match_path.exists():
        os.remove(match_path)
    logger.info(f"[{scene_name}] Removed intermediate files: {filtered_model_path} and {match_path}.")
    logger.info(f"Scene {scene_name} 3D triangulation on CPU completed.")

    # Step 4: Visualization
    logger.info(f"[{scene_name}] Generating 3D visualization...")
    visualize_sfm_3d(sfm_dir, scene_name, html_save_dir, True)


def main():
    parser = argparse.ArgumentParser(description="CPU Triangulation and Visualization")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Undistorted_SfM")
    parser.add_argument('--outputs', type=Path, required=True, help="Path to output sfm directory")
    parser.add_argument('--html_save_dir', type=Path, default=Path("./SfM_htmls"), help="Directory to save HTML visualizations")
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--scene_list', type=Path, default=None)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset}")
    args.html_save_dir.mkdir(parents=True, exist_ok=True)

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

    for scene_path in tqdm(scenes):
        try:
            process_scene(scene_path, args.outputs, args.html_save_dir)
        except Exception as e:
            logger.error(f"Failed to process {scene_path.name}: {e}")
            continue

if __name__ == "__main__":
    main()