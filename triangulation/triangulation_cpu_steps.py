from pathlib import Path
from visual_sfm_3d import visualize_sfm_3d
import warnings
import h5py
from hloc.utils import read_write_model as rw
from hloc import triangulation
import numpy as np
import shutil
import os
warnings.filterwarnings("ignore", category=FutureWarning)

def filter_reference_model(reference_model, output_model, valid_images):
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

root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
outputs = Path("/proj/vlarsson/outputs/sfm/")
html_save_dir =Path("/home/x_jiagu/degree_project/SfM_htmls")

scene_names = sorted([
    p.name
    for p in root.iterdir()
    if p.is_dir()
])

for scene in scene_names[:12]: # change the slice to process more scenes
    print(f"Start processing scene: {scene}...")

    images_path = root / scene / "images"

    output_dir = outputs / scene
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using CPU for processing 3D triangulation.")
    sfm_pairs = output_dir / "pairs-covisibility.txt"
    sfm_dir = output_dir / "sfm_superpoint+lightglue"
    feature_path = output_dir / "feats-superpoint-n2048.h5"
    match_path = output_dir / "feats-superpoint-n2048_matches-superpoint-lightglue_pairs-covisibility.h5"
    reference_model = root / scene / "sparse"

    # Step 1: Get valid images from matches
    with h5py.File(match_path, 'r') as f:
        groups = list(f.keys())
        valid_images = set(groups)
        for group in groups:
            valid_images.update(f[group].keys())

    print(f"Number of images with matches: {len(valid_images)}")

    # Step 2: Filter reference model, only keep images with matches
    filtered_model_path = output_dir / "sparse_filtered"
    filtered_model_path.mkdir(exist_ok=True)

    filter_reference_model(reference_model, filtered_model_path, valid_images)

    # Step 3: Triangulation to obtain 3D model
    model = triangulation.main(sfm_dir, filtered_model_path, images_path, sfm_pairs, feature_path, match_path)
    shutil.rmtree(filtered_model_path)
    os.remove(match_path)
    print(f"Removed intermediate files: {filtered_model_path} and {match_path}.")
    print(f"Secene {scene} 3D triangulation on CPU completed.")

    # Step 4: Visualization
    visualize_sfm_3d(sfm_dir, scene, html_save_dir, True)
