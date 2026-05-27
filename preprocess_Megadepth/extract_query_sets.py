import argparse
import logging
import random
from pathlib import Path
from tqdm import tqdm
from hloc.utils import read_write_model as rw

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random.seed(42)

def extract_query_sets(scene: str, sfm_model_dir: Path,depth_model_dir: Path, output_dir: Path,
                       sample_ratio=0.01, query_image_ratio=0.2):
    
    logger.info(f"Processing scene: {scene}")

    sfm_model_path = sfm_model_dir / scene / "sparse"
    if not (sfm_model_path / "cameras.bin").exists():
        if (sfm_model_dir / scene / "sparse" / "0").exists():
            sfm_model_path = sfm_model_dir / scene / "sparse" / "0"
    if not sfm_model_path.exists():
        logger.warning(f"Skipping {scene}: SfM path {sfm_model_path} not found.")
        return
    scene_depth_path = depth_model_dir / scene
    output_path = output_dir / scene
    output_path.mkdir(parents=True, exist_ok=True)

    # read the SfM model
    cameras, images, points3D = rw.read_model(sfm_model_path, ext=".bin")
    point_ids = list(points3D.keys())
    total_points = len(point_ids)
    num_samples = int(total_points * sample_ratio)
    sampled_ids = random.sample(point_ids, num_samples)
    logger.info(f"Points: {total_points} | Samples: {num_samples}")

    # check the depth image for better ground truth quality
    def has_depth(image_id):
        img_name = images[image_id].name
        h5_path1 = scene_depth_path / f"{Path(img_name).stem}.h5"
        h5_path2 = scene_depth_path / f"{img_name}.h5"
        return h5_path1.exists() or h5_path2.exists()
    
    # collect image names that observe the sampled 3D points
    observed_image_ids = set()
    query_image_ids = set()
    for pid in sampled_ids:
        img_ids_for_point = points3D[pid].image_ids
        observed_image_ids.update(img_ids_for_point)
        # filter out images without depth maps
        valid_img_ids = [iid for iid in img_ids_for_point if has_depth(iid)]
        
        if valid_img_ids:
            query_image_ids.update(random.sample(valid_img_ids, 1))

    logger.info(f"Total observed images: {len(observed_image_ids)}")

    # randomly select a subset of observed images as query images
    maximum_query_images = int(len(observed_image_ids) * query_image_ratio)
    if len(query_image_ids) > maximum_query_images:
        query_image_ids = random.sample(list(query_image_ids),  maximum_query_images)

    logger.info(f"Query images collected: {len(query_image_ids)}")

    query_infos = {}
    for img_id in query_image_ids:
        image = images[img_id]
        camera = cameras[image.camera_id]
        intrinsics = {
            "camera_id": camera.id,
            "model": camera.model,
            "width": camera.width,
            "height": camera.height,
            "params": camera.params.tolist()
        }
        pose = {
            "qvec": image.qvec.tolist(),
            "tvec": image.tvec.tolist()
        }

        query_infos[image.name] = {
            "intrinsics" : intrinsics,
            "pose" : pose
        }

    # write query image names and camera infos to files
    query_name_path = output_path / "query_image_names.txt"
    query_camera_path = output_path / "query_image_cameras.txt"

    with open(query_name_path, "w") as f_n, open(query_camera_path, "w") as f_c:
        for name, info in query_infos.items():

            f_n.write(f"{name}\n")

            qvec = info["pose"]["qvec"]
            tvec = info["pose"]["tvec"]
            intr = info["intrinsics"]

            line = (
                f"{name} "
                f"{' '.join(map(str, qvec))} "
                f"{' '.join(map(str, tvec))} "
                f"{intr['camera_id']} "
                f"{intr['model']} "
                f"{intr['width']} "
                f"{intr['height']} "
                f"{' '.join(map(str, intr['params']))}"
            )

            f_c.write(line + "\n")

    logger.info(f"Query image names written to: {query_name_path}")
    logger.info(f"Query image cameras written to: {query_camera_path}")

def main():

    parser = argparse.ArgumentParser(description="Extract query sets for 2D-3D matching.")
    parser.add_argument('--dataset', type=Path, default=Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM"))
    parser.add_argument('--outputs', type=Path, default=Path("/proj/vlarsson/outputs/query_sets"))
    parser.add_argument('--scene', type=str, default=None, help="Process a single scene ID, eg: 0000, 0001, etc.")
    parser.add_argument('--scene_list', type=Path, default=None, help="Path to a .txt file with scene IDs")
    parser.add_argument('--sample_ratio', type=float, default=0.0015)
    parser.add_argument('--query_ratio', type=float, default=0.20)
    args = parser.parse_args()

    depth_root = args.dataset.parent / "depth_undistorted"

    # scene selection
    if args.scene_list:
        with open(args.scene_list, 'r') as f:
            scenes = [line.strip() for line in f if line.strip()]
    elif args.scene:
        scenes = [args.scene]
    else:
        # or find all directories in dataset root
        scenes = sorted([p.name for p in args.dataset.iterdir() if p.is_dir()])

    logger.info(f"Starting extraction for {len(scenes)} scenes.")

    for scene in tqdm(scenes):
        try:
            extract_query_sets(scene, args.dataset, depth_root, args.outputs,
                               sample_ratio=args.sample_ratio, query_image_ratio=args.query_ratio)
        except Exception as e:
            logger.error(f"Failed to process scene {scene}: {e}")  

if __name__ == "__main__":
    main()