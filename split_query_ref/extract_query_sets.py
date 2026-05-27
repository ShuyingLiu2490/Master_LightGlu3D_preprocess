from hloc.utils import read_write_model as rw
import random
from pathlib import Path

random.seed(42)
def extract_query_sets(scene: str, 
                       sfm_model_dir: Path,
                       depth_model_dir: Path, 
                       output_dir: Path,
                       sample_ratio=0.01, 
                       query_image_ratio=0.2,
                       ):
    
    print(f"Processing scene: {scene}")

    sfm_model_path = sfm_model_dir / scene / "sparse"
    scene_depth_path = depth_model_dir / scene
    output_path = output_dir / scene
    output_path.mkdir(parents=True, exist_ok=True)

    # read the SfM model
    cameras, images, points3D = rw.read_model(sfm_model_path, ext=".bin")
    point_ids = list(points3D.keys())
    total_points = len(point_ids)
    num_samples = int(total_points * sample_ratio)
    sampled_ids = random.sample(point_ids, num_samples)
    print(f"Original 3D points: {total_points}; Sampled 3D points: {num_samples}")
    def has_depth(image_id):
        img_name = images[image_id].name
        h5_path1 = scene_depth_path / f"{Path(img_name).stem}.h5"
        h5_path2 = scene_depth_path / f"{img_name}.h5"
        return h5_path1.exists() or h5_path2.exists()
    
    # collect image names that observe the sampled 3D points
    oberved_image_ids = set()
    query_image_ids = set()
    for pid in sampled_ids:
        img_ids_for_point = points3D[pid].image_ids
        oberved_image_ids.update(img_ids_for_point)
        # filter out images without depth maps
        valid_img_ids = [iid for iid in img_ids_for_point if has_depth(iid)]
        
        if valid_img_ids:
            query_image_ids.update(random.sample(valid_img_ids, 1))

    print(f"Total oberved images: {len(oberved_image_ids)}")

    # randomly select a subset of oberved images as query images
    maximum_query_images = int(len(oberved_image_ids) * query_image_ratio)
    if len(query_image_ids) > maximum_query_images:
        query_image_ids = random.sample(list(query_image_ids),  maximum_query_images)

    print(f"Query images collected: {len(query_image_ids)}")

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

    print(f"Query image names written to: {query_name_path}")
    print(f"Query image cameras written to: {query_camera_path}")

if __name__ == "__main__":

    root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
    output_dir = Path("/proj/vlarsson/outputs")
    scene_names = sorted([p.name for p in root.iterdir() if p.is_dir()])

    for scene in scene_names[:13]:  # change the slice to process more scenes
        extract_query_sets(scene,
                           root,
                           root.parent / "depth_undistorted",
                           output_dir / "query_sets",
                           sample_ratio=0.0015,
                           query_image_ratio=0.20)
        