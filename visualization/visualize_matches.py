# For baseline: NN(Nearest Neighbour), RR(Rotate+Remove_coord), 
#               RN(Rotate+Normalize), PR(Project to Reference)
# For train: TRAIN(Lightglu3d two self and one bidirectional cross), 
#            ADAPT(lightglue+adapter)

# Before use it, add the gluefactory path in the terminal
# export PYTHONPATH="/home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match:$PYTHONPATH"

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
from utils.utils import qvec2rotmat, get_most_similar_ref
from . import rerun_johanna as rru 
from ground_truth.generate_gt_pairs_re import load_query_cams, compute_ground_truth_matches
from lightglue import LightGlue
from baseline.pr_baseline import compute_pr_baseline
import matplotlib.pyplot as plt
from lightglue import viz2d
from lightglue.utils import rbd
from baseline.rr_baseline import compute_rr_baseline, compute_precision_recall
from baseline.rn_baseline import compute_rn_baseline
from baseline.nn_baseline import compute_nn_baseline
from baseline.trained_matcher import load_trained_lightglu3d, load_trained_adapt, compute_trained_lightglu3d
from evaluation.pose_estimation_aachen import parse_aachen_cameras, load_reference_poses
from ground_truth.generate_ref_gt_pairs_from_hloc_aachen import compute_ground_truth_matches_aachen
from baseline.pr_baseline_change import compute_pr_baseline_change

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockCamera:
    def __init__(self, width, height, params):
        self.size = [width, height]
        self.f = [params[0], params[1]]
        self.c = [params[2], params[3]]

def get_image_path(dataset_path, dataset_type, scene, img_name, image_dir=None):
    """Helper to resolve image paths between MegaDepth and Aachen structures."""
    if dataset_type == "aachen":
        if image_dir is not None:
            p0 = image_dir / img_name
            if p0.exists(): return p0
            
        p1 = dataset_path / "images" / "images_upright" / img_name
        p2 = dataset_path / img_name
        
        return p1 if p1.exists() else p2
    else:
        return dataset_path / scene / "images" / img_name

def launch_rerun_visualization(pred_matches0, gt_matches0, q_kpts, p3d_kpts, raw_pts_np, raw_colors_np, scene, args, query_name, ref_name, camera, ref_pose_matrix, img_query_path, img_ref_path, dataset_type, method_name="Baseline", est_pose_matrix=None):    
    logger.info(f"Initializing Rerun Analytics Dashboard for {method_name}...")
    
    # Initialize rerun
    query_stem = Path(query_name).stem
    scene_label = scene if scene else "Aachen"
    rr.init(f"Matches_Scene_{scene_label}_{method_name}_{query_stem}", spawn=False)

    # Calculate matches info using the generic 'pred_matches0'
    valid_pred = pred_matches0 > -1
    valid_gt = gt_matches0 > -1
    
    idx_correct = valid_pred & valid_gt & (pred_matches0 == gt_matches0) # Correct prediction
    idx_confused = valid_pred & valid_gt & (pred_matches0 != gt_matches0) # Confused wrong prediction
    idx_hallucinated = valid_pred & (~valid_gt) # Hallucinated wrong prediction
    idx_missed = (~valid_pred) & valid_gt # Missed ground truth
    idx_unmatchable = (~valid_pred) & (~valid_gt) # Ignored unmatchable points

    # Set up cameras and images
    if dataset_type == "aachen":
        scene_sfm_dir = args.sfm_dir
    else:
        scene_sfm_dir = args.sfm_dir / scene if scene else args.sfm_dir
    cameras, images, _ = rw.read_model(scene_sfm_dir / "sfm_superpoint+lightglue", ext=".bin")
    ref_image_obj = next((img for img in images.values() if img.name == ref_name), None)
    ref_cam_obj = cameras[ref_image_obj.camera_id]
    ref_poselib_cam = MockCamera(ref_cam_obj.width, ref_cam_obj.height, ref_cam_obj.params)

    q_pose_matrix = np.hstack((qvec2rotmat(camera["qvec"]), np.array(camera["tvec"]).reshape(3, 1)))
    query_poselib_cam = MockCamera(camera["intrinsics"]["width"], camera["intrinsics"]["height"], camera["intrinsics"]["params"])

    # Use the resolved image paths passed into the function
    img_query = np.array(Image.open(img_query_path).convert("RGB")) / 255.0
    img_ref = np.array(Image.open(img_ref_path).convert("RGB")) / 255.0

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

    # Load query with categorized keypoints
    img_path = "world/camera_query_gt/image"
    rr.log(f"{img_path}/Correct", rr.Points2D(q_kpts[idx_correct], colors=[0, 255, 0], radii=4.0)) # Green
    rr.log(f"{img_path}/Confused", rr.Points2D(q_kpts[idx_confused], colors=[255, 165, 0], radii=3.0)) # Orange
    rr.log(f"{img_path}/Hallucinated", rr.Points2D(q_kpts[idx_hallucinated], colors=[255, 0, 0], radii=3.0)) # Red
    rr.log(f"{img_path}/Missed", rr.Points2D(q_kpts[idx_missed], colors=[0, 150, 255], radii=3.0)) # Blue
    rr.log(f"{img_path}/Unmatchable", rr.Points2D(q_kpts[idx_unmatchable], colors=[128, 0, 128], radii=2.0)) # Purple

    # Load visible sfm model
    rr.log("world/SfM_Context", rr.Points3D(raw_pts_np, colors=raw_colors_np, radii=0.03))
    cam_center = (-q_pose_matrix[:, :3].T @ q_pose_matrix[:, 3]).flatten()

    # Correct predicted matches (Green points and lines)
    correct_3d_pts = p3d_kpts[pred_matches0[idx_correct]]
    correct_lines = [[cam_center, pt] for pt in correct_3d_pts]
    rr.log("world/Predictions/Correct/Points", rr.Points3D(correct_3d_pts, colors=[0, 255, 0], radii=0.06))
    rr.log("world/Predictions/Correct/Lines", rr.LineStrips3D(correct_lines, colors=[0, 255, 0, 100]))
    
    # Confused wrong prediction (Orange points and lines + yellow error vectors)
    confused_3d_pts = p3d_kpts[pred_matches0[idx_confused]]
    confused_lines = [[cam_center, pt] for pt in confused_3d_pts]
    rr.log("world/Predictions/Confused/Points", rr.Points3D(confused_3d_pts, colors=[255, 165, 0], radii=0.06))
    rr.log("world/Predictions/Confused/Lines", rr.LineStrips3D(confused_lines, colors=[255, 165, 0, 80]))
    pred_pts_for_error = p3d_kpts[pred_matches0[idx_confused]]
    gt_pts_for_error = p3d_kpts[gt_matches0[idx_confused]]
    error_lines = [[gt_pt, pred_pt] for gt_pt, pred_pt in zip(gt_pts_for_error, pred_pts_for_error)]
    rr.log("world/Predictions/Confused/Error_Vectors", rr.LineStrips3D(error_lines, colors=[255, 255, 0, 200])) 

    # Hallucinated wrong prediction (Red points and lines)
    hallucinated_3d_pts = p3d_kpts[pred_matches0[idx_hallucinated]]
    hallucinated_lines = [[cam_center, pt] for pt in hallucinated_3d_pts]
    rr.log("world/Predictions/Hallucinated/Points", rr.Points3D(hallucinated_3d_pts, colors=[255, 0, 0], radii=0.06))
    rr.log("world/Predictions/Hallucinated/Lines", rr.LineStrips3D(hallucinated_lines, colors=[255, 0, 0, 80]))

    # Missed ground truth (Blue points and lines)
    missed_3d_pts = p3d_kpts[gt_matches0[idx_missed]]
    missed_lines = [[cam_center, pt] for pt in missed_3d_pts]
    rr.log("world/Ground_Truth/Missed/Points", rr.Points3D(missed_3d_pts, colors=[0, 150, 255], radii=0.06))
    rr.log("world/Ground_Truth/Missed/Lines", rr.LineStrips3D(missed_lines, colors=[0, 150, 255, 80]))

    # Save .rrd file
    output_filename = f"viz_{scene_label}_{method_name}_{query_stem}.rrd"
    rr.save(output_filename)
    logger.info(f"Rerun .rrd file visualization saved to {output_filename}")

def visual_flat_sfm(res, q_kpts, p3d_flat_kpts, img_query, p3d_colors, flat_w, flat_h, scene, method):
    logger.info(f"Generating 2D Flat SfM visualization for {method}...")
    
    # Create colored flat SfM image
    flat_img = np.ones((flat_h, flat_w, 3), dtype=np.float32)
    
    point_radius = 2
    # Paint the 3D point colors onto the 2D image
    for (x, y), c in zip(p3d_flat_kpts.astype(int), p3d_colors):
        # if 0 <= y < flat_h and 0 <= x < flat_w:
        #    flat_img[y, x] = c

        # Draw the image not outside of image, and change the radius to make the points more visible
        y_min = max(0, y - point_radius)
        y_max = min(flat_h, y + point_radius + 1)
        x_min = max(0, x - point_radius)
        x_max = min(flat_w, x + point_radius + 1)
        
        # Paint the block
        if y_min < y_max and x_min < x_max:
            flat_img[y_min:y_max, x_min:x_max] = c

    # Convert to format expected by LightGlue viz2d
    img_q_tensor = torch.from_numpy(img_query).float().permute(2, 0, 1)
    img_flat_tensor = torch.from_numpy(flat_img).float().permute(2, 0, 1)

    # Strip batch dimensions
    res_rbd = rbd(res)
    matches = res_rbd["matches"].cpu().numpy()
    
    m_kpts0 = q_kpts[matches[..., 0]]
    m_kpts1 = p3d_flat_kpts[matches[..., 1]]

    # The Matches plot
    viz2d.plot_images([img_q_tensor, img_flat_tensor])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {res_rbd["stop"]} layers', fs=20)

    # Lock bounds and prevent stretching
    axes = plt.gcf().axes
    if len(axes) >= 2:
        h0, w0 = img_query.shape[:2]
        axes[0].set_xlim(0, w0)
        axes[0].set_ylim(h0, 0)
        axes[1].set_xlim(0, flat_w)
        axes[1].set_ylim(flat_h, 0)

    # Add method to the filename
    scene_label = scene if scene else "Aachen"
    match_filename = f"flat_matches_scene_{scene_label}_{method}.png"
    plt.savefig(match_filename, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

    # The pruning plot
    if "prune0" in res_rbd:
        kpc0 = viz2d.cm_prune(res_rbd["prune0"])
        kpc1 = viz2d.cm_prune(res_rbd["prune1"])
        viz2d.plot_images([img_q_tensor, img_flat_tensor])
        viz2d.plot_keypoints([torch.from_numpy(q_kpts), torch.from_numpy(p3d_flat_kpts)], colors=[kpc0, kpc1], ps=6)
        
        # Lock bounds and prevent stretching
        axes = plt.gcf().axes
        if len(axes) >= 2:
            axes[0].set_xlim(0, w0)
            axes[0].set_ylim(h0, 0)
            axes[1].set_xlim(0, flat_w)
            axes[1].set_ylim(flat_h, 0)
            
        # Add method to the filename
        prune_filename = f"flat_pruning_scene_{scene_label}_{method}.png"
        plt.savefig(prune_filename, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
    logger.info(f"Saved 2D Flat SfM images: {match_filename} & {prune_filename}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Matches in Rerun")
    parser.add_argument('--dataset_type', type=str, default='megadepth', choices=['aachen', 'megadepth'])
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Undistorted_SfM or Aachen dataset root")
    parser.add_argument('--image_dir', type=Path, default=None, help="Explicit path to images directory (useful for Aachen unzipped images)")
    parser.add_argument('--covisibility_dir', type=Path, required=True, help="Path to covisibility")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query directory")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to sfm outputs")
    parser.add_argument('--depth_dir', type=Path, help="Path to depth maps (Required for MegaDepth)")
    parser.add_argument('--hloc_reference', type=Path, help="Path to hloc output (Required for Aachen GT)")
    parser.add_argument('--scene', type=str, default="", help="Scene name (or 'day'/'night' for Aachen)")
    parser.add_argument('--query_name', type=str, default=None, help="Optional: Specific query image name to visualize")
    parser.add_argument('--method', type=str, required=True, choices=['NN', 'RR', 'RN', 'PR', 'PRC', 'TRAIN', 'ADAPT'], 
                        help="Matching method to evaluate: NN, RR, RN, PR, PRC, TRAIN or ADAPT")
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help="Path to trained network weights (Only required if method is TRAIN or ADAPT)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = args.scene
    method = args.method
    scene_label = scene if scene else "Aachen"
    logger.info(f"Starting Evaluation & Visualization for {scene_label} using method: {method}")

    scene_query = args.query_dir / scene if (scene and args.query_dir and args.dataset_type != "aachen") else args.query_dir
    scene_covis = args.covisibility_dir / scene if (scene and args.dataset_type != "aachen") else args.covisibility_dir
    scene_sfm = args.sfm_dir / scene if (scene and args.dataset_type != "aachen") else args.sfm_dir

    # Load the list of available queries based on dataset type
    queries = []
    if args.dataset_type == "aachen":
        day_list_path = scene_query / "day_time_queries_with_intrinsics.txt"
        night_list_path = scene_query / "night_time_queries_with_intrinsics.txt"
        
        # Filter queries based on scene argument ('day' or 'night')
        if args.scene in ["day", ""] and day_list_path.exists():
            with open(day_list_path, 'r') as f:
                queries.extend([line.strip().split()[0] for line in f if line.strip() and not line.startswith("#")])
        if args.scene in ["night", ""] and night_list_path.exists():
            with open(night_list_path, 'r') as f:
                queries.extend([line.strip().split()[0] for line in f if line.strip() and not line.startswith("#")])
    else:
        query_names_file = scene_query / "query_image_names_clean.txt"
        with open(query_names_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]

    # Pick the query (specific or random)
    if args.query_name:
        if args.query_name not in queries:
            logger.error(f"Provided query '{args.query_name}' not found in the valid queries list.")
            return
        query_name = args.query_name
    else:
        query_name = random.choice(queries)

    ref_name = get_most_similar_ref(query_name, scene_covis / "most_similar_pairs.txt")
    logger.info(f"Query: {query_name} | Reference: {ref_name}")

    # Resolve image paths
    img_query_path = get_image_path(args.dataset, args.dataset_type, scene, query_name, args.image_dir)
    img_ref_path = get_image_path(args.dataset, args.dataset_type, scene, ref_name, args.image_dir)

    # Load query image
    img_query_pil = Image.open(img_query_path)
    q_img_size = np.array([img_query_pil.width, img_query_pil.height])

    # Load features
    with h5py.File(scene_sfm / "feats-superpoint-n2048.h5", "r") as f:
        q_kpts = f[query_name]["keypoints"][:]
        q_desc = f[query_name]["descriptors"][:]

    # Load visible 3d points
    sfm_model_path = scene_sfm / "sfm_superpoint+lightglue"
    reconstruction = pycolmap.Reconstruction(sfm_model_path)

    with open(scene_covis / "covisibility_results.pkl", "rb") as f:
        visible_p3d = pickle.load(f)[query_name]["unique_points"]

    p3d_desc, p3d_kpts, raw_colors = [], [], []

    with h5py.File(scene_covis / "points3D_feats_cache.h5", "r") as f:
        for pid in visible_p3d:
            pid_int = int(pid)
            pid_str = str(pid)
            if pid_str in f and pid_int in reconstruction.points3D:
                p3d_desc.append(f[pid_str]["descriptors"][:].reshape(256))
                p3d_kpts.append(f[pid_str]["keypoints"][:].reshape(3))
                raw_colors.append(reconstruction.points3D[pid_int].color)
    if not p3d_kpts:
        logger.error("No valid 3D coordinates/features found.")
        return

    p3d_desc = np.vstack(p3d_desc).T 
    p3d_kpts = np.vstack(p3d_kpts)   
    raw_pts_np = p3d_kpts.copy() 
    # Normalize colors for the flat image projection (0 to 1)
    raw_colors_np = np.vstack(raw_colors) / 255.0

    # Calculate ground truth dynamically based on dataset
    if args.dataset_type == "aachen":
        aachen_cams = parse_aachen_cameras(scene_query)
        gt_poses = load_reference_poses(args.hloc_reference)
        base_name = Path(query_name).name
        raw_camera = aachen_cams[query_name]
        gt_pose = gt_poses[base_name]
        camera = {
            "qvec": gt_pose["qvec"], "tvec": gt_pose["tvec"],
            "intrinsics": {
                "model": getattr(raw_camera, 'model_name', getattr(raw_camera.model, 'name', str(raw_camera.model))),
                "width": raw_camera.width, "height": raw_camera.height, "params": raw_camera.params
            }
        }
        gt_matches0, _ = compute_ground_truth_matches_aachen(
            q_kpts, p3d_kpts, raw_camera, gt_pose, pos_reproj_thresh=3.0, neg_reproj_thresh=5.0
        )
    else:
        query_cams = load_query_cams(scene_query / "query_image_cameras.txt")
        camera = query_cams[query_name]
        with h5py.File(args.depth_dir / scene / f"{Path(query_name).stem}.h5", 'r') as f:
            depth_map = f['depth'][:]
            
        gt_matches0, _ = compute_ground_truth_matches(
            {"keypoints": q_kpts}, {"keypoints": p3d_kpts}, camera, depth_map
        )

    # Load Reference Camera Pose from SfM
    cameras, images, _ = rw.read_model(sfm_model_path, ext=".bin")
    ref_image_obj = next((img for img in images.values() if img.name == ref_name), None)
    ref_R = qvec2rotmat(ref_image_obj.qvec)
    ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))
    ref_cam_obj = cameras[ref_image_obj.camera_id]

    # Initialize standard LightGlue for the projection baselines
    if method in ["RR", "RN", "PR", "PRC"]:
        logger.info("Initializing standard LightGlue for baseline evaluation...")
        baseline_matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    if method == "TRAIN":
        logger.info("Loaded trained LightGlu3D model for evaluation...")
        lightglu3d_matcher = load_trained_lightglu3d(args.checkpoint, device)
    if method == "ADAPT":
        logger.info("Loaded trained LightGlue_Adapt model for evaluation...")
        lightglu3_adapt_matcher = load_trained_adapt(args.checkpoint, device)

    # Get the predicted matches
    if method == "NN":
        pred_matches0 = compute_nn_baseline(q_desc, p3d_desc, device)
    elif method == "RR":
        pred_matches0, res, p3d_flat_kpts, flat_w, flat_h = compute_rr_baseline(
            baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device
        )
    elif method == "RN":
        pred_matches0, res, p3d_flat_kpts, flat_w, flat_h = compute_rn_baseline(
            baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, device
        )
    elif method == "PR":
        pred_matches0, res, p3d_flat_kpts, flat_w, flat_h = compute_pr_baseline(
            baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, ref_cam_obj, device
        )
    elif method == "PRC":
        pred_matches0, res, p3d_flat_kpts, flat_w, flat_h = compute_pr_baseline_change(
            baseline_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, ref_pose_matrix, camera, device
        )
    elif method == "TRAIN":
        if args.checkpoint is None:
            raise ValueError("--checkpoint must be provided when using the TRAIN method.")
        pred_matches0, _ = compute_trained_lightglu3d(lightglu3d_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)
    elif method == "ADAPT":
        if args.checkpoint is None:
            raise ValueError("--checkpoint must be provided when using the ADAPT method.")
        pred_matches0, _ = compute_trained_lightglu3d(lightglu3_adapt_matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device)
    
    # Evaluate metrics
    precision, recall, num_gt, num_pred, num_correct = compute_precision_recall(pred_matches0, gt_matches0)

    # Handle None values gracefully for formatting
    disp_precision = precision if precision is not None else 0.0
    disp_recall = recall if recall is not None else 0.0

    logger.info("="*30)
    logger.info(f"{method} Results:")
    logger.info(f"GT Matches:        {num_gt}")
    logger.info(f"Predicted Matches: {num_pred}")
    logger.info(f"Correct Matches:   {num_correct}")
    logger.info(f"Precision:         {disp_precision:.4f}")
    logger.info(f"Recall:            {disp_recall:.4f}")
    logger.info("="*30)

    # Launch 2D flat images
    if method in ["RR", "RN", "PR", "PRC"]:
        img_query_np = np.array(img_query_pil.convert("RGB")) / 255.0
        visual_flat_sfm(res, q_kpts, p3d_flat_kpts, img_query_np, raw_colors_np, flat_w, flat_h, scene, method)

    # Launch rerun
    launch_rerun_visualization(
        pred_matches0=pred_matches0, 
        gt_matches0=gt_matches0,  
        q_kpts=q_kpts, p3d_kpts=p3d_kpts, 
        raw_pts_np=raw_pts_np, raw_colors_np=raw_colors_np,
        scene=scene, args=args, query_name=query_name, ref_name=ref_name, 
        camera=camera,
        ref_pose_matrix=ref_pose_matrix,
        img_query_path=img_query_path,
        img_ref_path=img_ref_path,
        dataset_type=args.dataset_type,
        method_name=method
    )

if __name__ == "__main__":
    main()