import argparse
import logging
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from hloc import extract_features, pairs_from_retrieval
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat
from preprocess_Megadepth.covisibility_search_pipe import most_similar_pair, covisibility_search

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def map_img_name_to_id(img_name, images):
    for img_id, img in images.items():
        if img.name == img_name:
            return img_id
    return None

def map_img_to_points3d(img_name, images):
    img_id = map_img_name_to_id(img_name, images)
    if img_id is None:
        return np.array([])
    p3d_ids = images[img_id].point3D_ids
    return p3d_ids[p3d_ids != -1]

def get_aachen_queries(query_dir):
    queries = []
    query_files = [
        query_dir / "day_time_queries_with_intrinsics.txt",
        query_dir / "night_time_queries_with_intrinsics.txt"
    ]
    
    for query_file in query_files:
        if query_file.exists():
            with open(query_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        queries.append(line.strip().split()[0])
    return queries

def most_similar_pair(image_dir, output_dir, query_list_path, sfm_model_path):
    # Chnage the path for Aachen
    ref_list = output_dir / "reference_list.txt"
    references_features = output_dir / 'feats-netvlad-ref.h5'
    queries_features = output_dir / 'feats-netvlad-query.h5'
    pair_file = output_dir / "most_similar_pairs.txt"

    _, images, _ = rw.read_model(sfm_model_path, ext=".bin")
    image_names = [img.name for img in images.values()]
    
    logger.info("Writing reference list (database images with 3D points)...")
    with open(ref_list, "w") as f:
        for name in sorted(image_names):
            if len(map_img_to_points3d(name, images)) == 0:
                continue # Skip images with no 3D correspondences in SfM model.
            f.write(str(name) + "\n")

    feature_conf = extract_features.confs["netvlad"]
    
    logger.info("Extracting NetVLAD global features for REFERENCE images...")
    extract_features.main(
        conf=feature_conf,
        image_list=ref_list,
        image_dir=image_dir,
        feature_path=references_features,
    )

    logger.info("Extracting NetVLAD global features for QUERY images...")
    extract_features.main(
        conf=feature_conf,
        image_list=query_list_path,
        image_dir=image_dir,
        feature_path=queries_features,
    )

    logger.info("Performing image retrieval (NetVLAD matching)...")
    pairs_from_retrieval.main(
        descriptors=queries_features,
        db_descriptors=references_features,     
        output=pair_file,               
        num_matched=1, # could change this for more similar images
        query_list=query_list_path,        
        db_list=ref_list,            
    )

    matched_pairs_dict = defaultdict(list)
    with open(pair_file) as f:
        for line in f.readlines():
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    matched_pairs_dict[parts[0]].append(parts[1])

    return matched_pairs_dict

def process_aachen(args):
    logger.info("Starting Covisibility Search for Aachen Dataset...")
    
    output_dir = args.outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    sfm_model_path = args.sfm_dir / "sfm_superpoint+lightglue"
    if not sfm_model_path.exists():
        raise FileNotFoundError(f"SfM model not found at {sfm_model_path}")
        
    # Parse query files and create clean_aachen_queries.txt
    queries = get_aachen_queries(args.query_dir)
    query_list_path = output_dir / "clean_aachen_queries.txt"
    with open(query_list_path, 'w') as f:
        for q in queries:
            f.write(f"{q}\n")
    logger.info(f"Extracted {len(queries)} query images to {query_list_path}")

    # Find the most similar images via NetVLAD
    matched_pairs_dict = most_similar_pair(
        image_dir=args.image_dir,
        output_dir=output_dir,
        query_list_path=query_list_path,
        sfm_model_path=sfm_model_path
    )

    # Load SfM model for covisibility graph
    logger.info("Loading SfM model for Covisibility Search...")
    _, images, point3D = rw.read_model(sfm_model_path, ext=".bin")

    # Conduct covisibility search
    covisibility_results = {}

    # Separate lists for Day and Night point counts
    point_counts_day = [] 
    point_counts_night = []

    # point_counts = [] 
    log_file_path = output_dir / "query_details.txt"
    
    with open(log_file_path, "w") as log_f:
        for query_image, matched_images in tqdm(matched_pairs_dict.items(), desc="Running Covisibility Search"):
            for matched_image in matched_images:
                points3d_level = map_img_to_points3d(matched_image, images)
                if len(points3d_level) != 0: 
                    break

            if len(points3d_level) == 0:
                logger.warning(f"No 3D points found for reference image {matched_image}. Skipping query {query_image}.")
                continue

            img = images[map_img_name_to_id(matched_image, images)]
            R, t = qvec2rotmat(img.qvec), img.tvec
            camera_center = -R.T @ t   

            unique_images, unique_points, max_distance = covisibility_search(
                points3d_level=points3d_level,
                images=images,
                points3D=point3D,
                camera_pos=camera_center,
                pruning=args.pruning,
                max_points=args.max_points
            )
            
            # point_counts.append(len(unique_points))
            if "day" in query_image.lower():
                point_counts_day.append(len(unique_points))
            else:
                point_counts_night.append(len(unique_points))

            # Write logs
            log_f.write(f"Query Image: {query_image}, Matched Image: {matched_image}\n")
            log_f.write(f"  Unique Images Found: {len(unique_images)}\n")
            log_f.write(f"  Unique 3D Points Found: {len(unique_points)}\n")
            log_f.write(f"  Max Camera Distance: {max_distance:.2f}\n\n")
            
            covisibility_results[query_image] = {
                'unique_images': unique_images,
                'unique_points': unique_points,
                'max_distance': max_distance
            }

    # Print summary
    # if point_counts:
    #     logger.info("="*40)
    #     logger.info("Covisibility Summary for Aachen")
    #     logger.info(f"Target Point Limitation: {args.max_points}")
    #     logger.info(f"Smallest 3D Pointcloud:  {np.min(point_counts)}")
    #     logger.info(f"Largest 3D Pointcloud:   {np.max(point_counts)}")
    #     logger.info(f"Average 3D Pointcloud:   {np.mean(point_counts):.2f}")
    #     logger.info(f"(Detailed results saved to: {log_file_path})")
    #     logger.info("="*40)

    if point_counts_day or point_counts_night:
        plt.figure(figsize=(10, 5))
        bins = np.linspace(0, args.max_points, 50)
        
        if point_counts_day:
            plt.hist(point_counts_day, bins=bins, alpha=0.6, color='orange', label='Aachen Day', density=True)
        if point_counts_night:
            plt.hist(point_counts_night, bins=bins, alpha=0.6, color='blue', label='Aachen Night', density=True)
            
        plt.title('Aachen Day vs Night: 3D Point Covisibility Distribution')
        plt.xlabel('Number of Visible 3D Points')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plot_path = "/home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/covisibility/aachen_day_night_distribution.png"
        plt.savefig(plot_path)
        logger.info(f"Saved distribution plot to {plot_path}")

    # Save covisibility_results.pkl
    with open(output_dir / "covisibility_results.pkl", "wb") as f:
        pickle.dump(covisibility_results, f)
    
def main():
    parser = argparse.ArgumentParser(description="Covisibility Search Pipeline for Aachen")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Aachen root")
    parser.add_argument('--image_dir', type=Path, required=True, help="Path to images_upright")
    parser.add_argument('--outputs', type=Path, required=True, help="Path to save covisibility outputs")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to the directory containing triangulated models")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to original queries directory")
    parser.add_argument('--pruning', type=float, default=0.35, help="Pruning factor for covisibility")
    parser.add_argument('--max_points', type=int, default=8192, help="Max unique 3D points")
    args = parser.parse_args()
    
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset}")

    try:
        process_aachen(args)
    except Exception as e:
        logger.error(f"Failed to process Aachen covisibility search: {e}", exc_info=True)

if __name__ == "__main__":
    main()