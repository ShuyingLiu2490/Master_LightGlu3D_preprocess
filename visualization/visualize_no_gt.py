import argparse
import logging
import pickle
import random
import numpy as np
import torch
import h5py
import rerun as rr
from pathlib import Path
from PIL import Image
import pycolmap
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat
from . import rerun_johanna as rru 
from ground_truth.generate_gt_pairs_by_scene import load_query_cams, compute_ground_truth_matches
from .visualize_matches import get_most_similar_ref, MockCamera

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Find and Visualize GT=0 Cases in Rerun")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Undistorted_SfM")
    parser.add_argument('--covisibility_dir', type=Path, required=True, help="Path to covisibility")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to sfm outputs")
    parser.add_argument('--depth_dir', type=Path, required=True, help="Path to depth maps")
    parser.add_argument('--scene', type=str, required=True)
    args = parser.parse_args()

    scene = args.scene
    logger.info(f"Searching for a GT=0 example in Scene {scene}...")

    # Load resources that are common to all queries
    sfm_model_path = args.sfm_dir / scene / "sfm_superpoint+lightglue"
    reconstruction = pycolmap.Reconstruction(sfm_model_path)
    cameras, images, _ = rw.read_model(sfm_model_path, ext=".bin")
    
    with open(args.covisibility_dir / scene / "covisibility_results.pkl", "rb") as f:
        covis_dict = pickle.load(f)
        
    query_cams = load_query_cams(args.query_dir / scene / "query_image_cameras.txt")

    # Randomly shuffle the list of queries
    query_names_file = args.query_dir / scene / "query_image_names.txt"
    with open(query_names_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    random.shuffle(queries)

    # Variables to hold our chosen data
    target_query = None
    target_p3d_kpts = None
    target_raw_colors = None
    target_q_kpts = None

    # Loop until find a GT = 0 match
    for query_name in queries:
        if query_name not in covis_dict:
            continue
            
        visible_p3d = covis_dict[query_name]["unique_points"]
        if len(visible_p3d) == 0:
            continue # Skip if no 3D points exist at all
            
        # Load 2D keypoints
        with h5py.File(args.sfm_dir / scene / "feats-superpoint-n2048.h5", "r") as f:
            if query_name not in f: continue
            q_kpts = f[query_name]["keypoints"][:]
            
        # Load 3D keypoints
        p3d_kpts = []
        raw_colors = []
        with h5py.File(args.covisibility_dir / scene / "points3D_feats_cache.h5", "r") as f:
            for pid in visible_p3d:
                pid_int, pid_str = int(pid), str(pid)
                if pid_str in f and pid_int in reconstruction.points3D:
                    p3d_kpts.append(f[pid_str]["keypoints"][:].reshape(3))
                    raw_colors.append(reconstruction.points3D[pid_int].color)
                    
        if len(p3d_kpts) == 0:
            continue

        p3d_kpts = np.vstack(p3d_kpts)   
        
        # Load depth map
        depth_file = args.depth_dir / scene / f"{Path(query_name).stem}.h5"
        if not depth_file.exists():
            continue
            
        with h5py.File(depth_file, 'r') as f:
            depth_map = f['depth'][:]
            
        # Compute ground truth
        q_camera = query_cams[query_name]
        gt_matches0, _ = compute_ground_truth_matches(
            {"keypoints": q_kpts}, {"keypoints": p3d_kpts}, q_camera, depth_map
        )
        
        valid_gt = gt_matches0 > -1
        num_gt = valid_gt.sum()
        
        # Found a gt = 0 case with covisible points
        if num_gt == 0:
            logger.info(f"Found query with GT=0: {query_name} (covisible 3D points: {len(p3d_kpts)})")
            target_query = query_name
            target_p3d_kpts = p3d_kpts
            target_raw_colors = np.vstack(raw_colors) / 255.0
            target_q_kpts = q_kpts
            break

    if target_query is None:
        logger.error("Could not find any query with exactly 0 GT matches and >0 covisible points.")
        return

    # Setup rerun
    ref_name = get_most_similar_ref(target_query, args.covisibility_dir / scene / "most_similar_pairs.txt")
    
    # Load reference camera pose
    ref_image_obj = next((img for img in images.values() if img.name == ref_name), None)
    ref_R = qvec2rotmat(ref_image_obj.qvec)
    ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))
    ref_cam_obj = cameras[ref_image_obj.camera_id]
    ref_poselib_cam = MockCamera(ref_cam_obj.width, ref_cam_obj.height, ref_cam_obj.params)

    # Load query camera pose
    camera = query_cams[target_query]
    q_pose_matrix = np.hstack((qvec2rotmat(camera["qvec"]), np.array(camera["tvec"]).reshape(3, 1)))
    query_poselib_cam = MockCamera(camera["intrinsics"]["width"], camera["intrinsics"]["height"], camera["intrinsics"]["params"])

    # Load images
    img_query = np.array(Image.open(args.dataset / scene / "images" / target_query).convert("RGB")) / 255.0
    img_ref = np.array(Image.open(args.dataset / scene / "images" / ref_name).convert("RGB")) / 255.0

    logger.info(f"Initializing rerun for GT=0 analysis...")
    rr.init(f"GT_Zero_Scene_{scene}", spawn=False)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # Plot scene
    rru.plot_scene(
        pts_3d=np.empty((0,3)), pts_2d=np.empty((0,2)),           
        img_query=img_query, imgs_refs=[img_ref], 
        camera_poses_refs=np.array([ref_pose_matrix]), 
        poselib_cam_intrinsics_q=query_poselib_cam,
        poselib_cam_intrinsics_refs=[ref_poselib_cam], 
        cam_pose_query_estimated=None,  
        cam_pose_query_gt=q_pose_matrix, 
        attach_image_to_est_pose=False  
    )

    # Log the 3D points and 2D keypoints
    rr.log("world/SfM_Context", rr.Points3D(target_p3d_kpts, colors=target_raw_colors, radii=0.03))
    rr.log("world/camera_query_gt/image/Keypoints_2D", rr.Points2D(target_q_kpts, colors=[255, 255, 0], radii=3.0)) # yellow 2d keypoints

    output_filename = f"visualization_scene_{scene}_GT_ZERO.rrd"
    rr.save(output_filename)
    logger.info(f"Rerun .rrd file visualization saved to {output_filename}")

if __name__ == "__main__":
    main()