import numpy as np
from collections import defaultdict
from hloc import extract_features, pairs_from_retrieval
from hloc.utils import read_write_model as rw
from pathlib import Path
import pickle
from utils import map_img_to_points3d, map_img_name_to_id, qvec2rotmat

def most_similar_pair(reference_dir, query_dir, output_dir):
    '''
    Finds the most similar image in the reference directory for each query in the query dir.
    '''
    ref_list = output_dir / "reference_list.txt"
    query_list = output_dir.parent.parent / "query_sets" / output_dir.name / "query_image_names.txt"
    # query_list = output_dir / "query_list.txt"
    # TODO: training stage read all queries from dataset, uncomment this when inference

    references_features = output_dir / 'feats-netvlad-ref.h5'
    queries_features = output_dir / 'feats-netvlad-query.h5'
    pair_file = output_dir / "most_similar_pairs.txt"

    _, images, _ = rw.read_model(
        output_dir.parent.parent / "sfm" / output_dir.name / "sfm_superpoint+lightglue", ext=".bin"
        )
    image_names = [img.name for img in images.values()]
    with open(ref_list, "w") as f:
        for name in sorted(image_names):
            if len(map_img_to_points3d(name, images)) == 0:
                continue # Skip images with no 3D correspondences in SfM model.
            f.write(str(name) + "\n")

    # with open(ref_list, "w") as f:
    #    # TODO: training stage read all references from .bin, uncomment this when inference
    #     for img in sorted(reference_dir.iterdir()):
    #         if img.suffix.lower() in [".jpg", ".png", ".jpeg"] and img.stat().st_size != 0:
    #             f.write(str(img.name) + "\n")

    # with open(query_list, "w") as f:
    #     # TODO: training stage read all queries from dataset, uncomment this when inference
    #     for query in sorted(query_dir.iterdir()):
    #         if query.suffix.lower() in [".jpg", ".png", ".jpeg"]:
    #             f.write(str(query.name) + "\n")

    feature_conf = extract_features.confs["netvlad"]
    print("Extracting global features...")

    extract_features.main(
        conf=feature_conf,
        image_list=ref_list,
        image_dir= reference_dir,
        feature_path=references_features,
    )

    extract_features.main(
        conf=feature_conf,
        image_list=query_list,
        image_dir=query_dir,
        feature_path=queries_features,
    )

    print("Performing image retrieval...")

    # pairs_from_retrieval.main(
    #     descriptors=feature_dir,     
    #     output=pair_file,               
    #     num_matched=5,   # find more similar images, in case the most similar one has no 3D points in SfM model.                 
    #     query_list=query_list,        
    #     db_list=ref_list,            
    # )

    pairs_from_retrieval.main(
        descriptors=queries_features,
        db_descriptors=references_features,     
        output=pair_file,               
        num_matched=1,                   
        query_list=query_list,        
        db_list=ref_list,            
    )

    print("\nThe most similar reference image:")
    matched_pairs_dict = defaultdict(list)
    with open(pair_file) as f:
        for line in f.readlines():
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    matched_pairs_dict[parts[0]].append(parts[1])
                    # print(f"Query Image: {parts[0]}  -->  Matched Image: {parts[1]}")
                else:
                    print(f"Invalid line format: {line.strip()}")

    return matched_pairs_dict

def covisibility_search(
    points3d_level: np.ndarray,
    images: dict,
    points3D: dict,
    camera_pos: np.ndarray = np.array([0,0,0]),
    pruning: float = 0.5,
    max_points: int = 10000,
) -> tuple:
    """
    Conducts covisibility search through bipartite PR.
    Args:
        points3d_level: Array of 3D point IDs to start search.
        images: Dictionary of image data from SfM model.
        points3D: Dictionary of 3D point data from SfM model.
        camera_pos: Camera position as a numpy array.
        pruning: Pruning factor for covisibility search.
    Returns:
        valid_images: Set of image IDs where points are visible after pruning.
        unique_points: Array of unique 3D point IDs found.
        max_distance: Maximum camera distance found.
    """
    unique_images = set()
    unique_points = set(points3d_level)
    max_distance=0.0

    if len(points3d_level) == 0:
        return set(), np.array([]), 0.0
    
    for pid in points3d_level:
        if pid in points3D:
            img_ids = points3D[pid].image_ids
            img_ids = img_ids[img_ids != -1]
            unique_images.update(img_ids)

    # # ORIGINAL algorithm
    # # Ensure we are not including images with too small overlap
    # valid_images = set()
    # for ind in unique_images:
    #     img=images[ind]
    #     points_3d_image=img.point3D_ids[np.where(img.point3D_ids!=-1)]
    #     intersection=set(points_3d_image).intersection(points3d_level)
    #     if len(intersection)/len(points3d_level) > pruning:
    #         R, t= qvec2rotmat(img.qvec), img.tvec
    #         C=-R.T@t
    #         distance=np.linalg.norm(np.array(camera_pos) - np.array(C))
    #         if distance>max_distance:
    #             max_distance=distance
    #         unique_points.update(points_3d_image)
    #         valid_images.add(ind)
    #         if len(unique_points) > max_points:
    #             # Limit the number of unique points to 10000 for efficiency
    #             unique_points=set(list(unique_points)[:max_points])
    #             break

    # A new algorithm
    valid_images = set()
    image_scores = []
    point3d_dict = {}
    for ind in unique_images:
        img=images[ind]
        points_3d_image=img.point3D_ids[np.where(img.point3D_ids!=-1)]
        intersection=set(points_3d_image).intersection(points3d_level)
        image_scores.append((len(intersection)/len(points3d_level), ind))
        point3d_dict[ind] = points_3d_image
    # Sort all candidate images by overlap scores: high -> low.
    # First save the visible 3D points from high-score images until we get MAX NUMBER point3d.
    # But if the overlap is too small, break in advance
    image_scores.sort(key=lambda x: x[0], reverse=True)
    for overlap_score, im_id in image_scores:
        # Ensure we are not including images with too small overlap
        if overlap_score <= pruning: 
            break 

        valid_images.add(im_id)
        unique_points.update(point3d_dict[im_id])
        img = images[im_id]
        R, t= qvec2rotmat(img.qvec), img.tvec
        C=-R.T@t
        distance=np.linalg.norm(np.array(camera_pos) - np.array(C))
        if distance>max_distance:
            max_distance=distance
            
        if len(unique_points) >= max_points:
            # Limit the number of unique points to 10000 for efficiency
            expansion_points = list(unique_points - set(points3d_level))
            buffer = max_points - len(points3d_level)
            if buffer > 0:
                unique_points = list(points3d_level) + expansion_points[:buffer]
            else:
                unique_points = list(points3d_level)[:max_points]
            
            unique_points=np.array(list(unique_points))
            return valid_images, unique_points[np.where(unique_points!=-1)], max_distance

       
    unique_points=np.array(list(unique_points))    
    unique_points=unique_points[np.where(unique_points!=-1)] # Remove points not in sfm model.

    return valid_images, unique_points, max_distance



if __name__ == "__main__":

    root = Path("/proj/vlarsson/datasets/megadepth/Undistorted_SfM")
    # outputs = Path("/proj/vlarsson/outputs/sfm/")
    outputs = Path("/proj/vlarsson/outputs/midterm_results/")
    scene_names = sorted([
        p.name
        for p in root.iterdir()
        if p.is_dir()
    ])

    for scene in scene_names[:12]: # Change the slice to process more scenes
        print(f"Start processing covisibility search for scene: {scene}...")
        images_path = root / scene / "images" # Contains all .jpg images
        output_dir = outputs / scene
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find the most similar images for each query image in Scenexxxx
        matched_pairs_dict = most_similar_pair(
            reference_dir=images_path,
            query_dir=images_path,
            output_dir=output_dir
        )

        # Load SfM model
        _, images, point3D = rw.read_model(
            output_dir.parent.parent / "sfm" / output_dir.name / "sfm_superpoint+lightglue", ext=".bin"
            )
        
        # Conduct covisibility search for each matched pair
        covisibility_results = {}
        for query_image, matched_images in matched_pairs_dict.items():
            for matched_image in matched_images:
                points3d_level = map_img_to_points3d(matched_image, images)
                if len(points3d_level) != 0: 
                    # If the most similar image has no 3D correspondences, move to the next one
                    break

            img = images[map_img_name_to_id(matched_image, images)]
            R, t = qvec2rotmat(img.qvec), img.tvec
            camera_center = -R.T @ t   

            unique_images, unique_points, max_distance = covisibility_search(
                points3d_level=points3d_level,
                images=images,
                points3D=point3D,
                camera_pos=camera_center,
                pruning=0.35,
                max_points=8192
            )
            print(f"Query Image: {query_image}, Matched Image: {matched_image}")
            print(f"  Unique Images Found: {len(unique_images)}")
            print(f"  Unique 3D Points Found: {len(unique_points)}")
            print(f"  Max Camera Distance: {max_distance:.2f}\n")
            covisibility_results[query_image] = {
                'unique_images': unique_images,
                'unique_points': unique_points,
                'max_distance': max_distance
            }
        print(f"Finished covisibility search for scene: {scene}.")

        # Save covisibility_results
        with open(output_dir / "covisibility_results.pkl", "wb") as f:
            pickle.dump(covisibility_results, f)