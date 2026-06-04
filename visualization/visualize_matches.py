import argparse
import logging
import random
import numpy as np
import torch
import rerun as rr
from pathlib import Path
from PIL import Image
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat
from utils import rerun_johanna as rru 
from utils.matching_performance import compute_precision_recall
from preprocess_Megadepth.dataloader import MegaDepthLoader
from lightglue import LightGlue
import matplotlib.pyplot as plt
from lightglue import viz2d
from lightglue.utils import rbd
from baselines_and_trained_matcher.mnn_baseline import compute_nn_baseline
from baselines_and_trained_matcher.trained_matcher import load_trained_lightglu3d, compute_trained_lightglu3d
from baselines_and_trained_matcher.pr_lg_baseline import compute_pr_baseline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockCamera:
    def __init__(self, width, height, params):
        self.size = [width, height]
        self.f = [params[0], params[1]]
        self.c = [params[2], params[3]]

def launch_rerun_visualization(pred_matches0, data, scene, args, method_name="Baseline"):    
    logger.info(f"Initializing Rerun Analytics Dashboard for {method_name}...")
    
    query_stem = Path(data["query_name"]).stem
    rr.init(f"Matches_Scene_{scene}_{method_name}_{query_stem}", spawn=False)

    valid_pred = pred_matches0 > -1
    valid_gt = data["gt_matches0"] > -1
    
    idx_correct = valid_pred & valid_gt & (pred_matches0 == data["gt_matches0"])
    idx_confused = valid_pred & valid_gt & (pred_matches0 != data["gt_matches0"])
    idx_hallucinated = valid_pred & (~valid_gt) 
    idx_missed = (~valid_pred) & valid_gt 
    idx_unmatchable = (~valid_pred) & (~valid_gt) 

    scene_sfm_dir = args.sfm_dir / scene
    cameras, images, _ = rw.read_model(scene_sfm_dir / "sfm_superpoint+lightglue", ext=".bin")
    ref_image_obj = next((img for img in images.values() if img.name == data["ref_name"]), None)
    ref_cam_obj = cameras[ref_image_obj.camera_id]
    ref_poselib_cam = MockCamera(ref_cam_obj.width, ref_cam_obj.height, ref_cam_obj.params)

    q_pose_matrix = np.hstack((qvec2rotmat(data["camera"]["qvec"]), np.array(data["camera"]["tvec"]).reshape(3, 1)))
    query_poselib_cam = MockCamera(data["camera"]["intrinsics"]["width"], data["camera"]["intrinsics"]["height"], data["camera"]["intrinsics"]["params"])

    img_query = np.array(Image.open(data["img_query_path"]).convert("RGB")) / 255.0
    img_ref = np.array(Image.open(data["img_ref_path"]).convert("RGB")) / 255.0

    rru.plot_scene(
        pts_3d=np.empty((0,3)), pts_2d=np.empty((0,2)),           
        img_query=img_query, imgs_refs=[img_ref], 
        camera_poses_refs=np.array([data["ref_pose_matrix"]]), 
        poselib_cam_intrinsics_q=query_poselib_cam,
        poselib_cam_intrinsics_refs=[ref_poselib_cam], 
        cam_pose_query_estimated=None,  
        cam_pose_query_gt=q_pose_matrix, 
        attach_image_to_est_pose=False  
    )

    img_path = "world/camera_query_gt/image"
    rr.log(f"{img_path}/Correct", rr.Points2D(data["q_kpts"][idx_correct], colors=[0, 255, 0], radii=4.0)) 
    rr.log(f"{img_path}/Confused", rr.Points2D(data["q_kpts"][idx_confused], colors=[255, 165, 0], radii=3.0)) 
    rr.log(f"{img_path}/Hallucinated", rr.Points2D(data["q_kpts"][idx_hallucinated], colors=[255, 0, 0], radii=3.0)) 
    rr.log(f"{img_path}/Missed", rr.Points2D(data["q_kpts"][idx_missed], colors=[0, 150, 255], radii=3.0)) 
    rr.log(f"{img_path}/Unmatchable", rr.Points2D(data["q_kpts"][idx_unmatchable], colors=[128, 0, 128], radii=2.0)) 

    rr.log("world/SfM_Context", rr.Points3D(data["raw_pts_np"], colors=data["raw_colors_np"], radii=0.03))
    cam_center = (-q_pose_matrix[:, :3].T @ q_pose_matrix[:, 3]).flatten()

    correct_3d_pts = data["p3d_kpts"][pred_matches0[idx_correct]]
    correct_lines = [[cam_center, pt] for pt in correct_3d_pts]
    rr.log("world/Predictions/Correct/Points", rr.Points3D(correct_3d_pts, colors=[0, 255, 0], radii=0.06))
    rr.log("world/Predictions/Correct/Lines", rr.LineStrips3D(correct_lines, colors=[0, 255, 0, 100]))
    
    confused_3d_pts = data["p3d_kpts"][pred_matches0[idx_confused]]
    confused_lines = [[cam_center, pt] for pt in confused_3d_pts]
    rr.log("world/Predictions/Confused/Points", rr.Points3D(confused_3d_pts, colors=[255, 165, 0], radii=0.06))
    rr.log("world/Predictions/Confused/Lines", rr.LineStrips3D(confused_lines, colors=[255, 165, 0, 80]))
    
    pred_pts_for_error = data["p3d_kpts"][pred_matches0[idx_confused]]
    gt_pts_for_error = data["p3d_kpts"][data["gt_matches0"][idx_confused]]
    error_lines = [[gt_pt, pred_pt] for gt_pt, pred_pt in zip(gt_pts_for_error, pred_pts_for_error)]
    rr.log("world/Predictions/Confused/Error_Vectors", rr.LineStrips3D(error_lines, colors=[255, 255, 0, 200])) 

    hallucinated_3d_pts = data["p3d_kpts"][pred_matches0[idx_hallucinated]]
    hallucinated_lines = [[cam_center, pt] for pt in hallucinated_3d_pts]
    rr.log("world/Predictions/Hallucinated/Points", rr.Points3D(hallucinated_3d_pts, colors=[255, 0, 0], radii=0.06))
    rr.log("world/Predictions/Hallucinated/Lines", rr.LineStrips3D(hallucinated_lines, colors=[255, 0, 0, 80]))

    missed_3d_pts = data["p3d_kpts"][data["gt_matches0"][idx_missed]]
    missed_lines = [[cam_center, pt] for pt in missed_3d_pts]
    rr.log("world/Ground_Truth/Missed/Points", rr.Points3D(missed_3d_pts, colors=[0, 150, 255], radii=0.06))
    rr.log("world/Ground_Truth/Missed/Lines", rr.LineStrips3D(missed_lines, colors=[0, 150, 255, 80]))

    output_filename = f"viz_{scene}_{method_name}_{query_stem}.rrd"
    rr.save(output_filename)
    logger.info(f"Rerun .rrd file visualization saved to {output_filename}")

def visual_flat_sfm(res, data, flat_w, flat_h, scene, method):
    logger.info(f"Generating 2D Flat SfM visualization for {method}...")
    
    flat_img = np.ones((flat_h, flat_w, 3), dtype=np.float32)
    point_radius = 2
    
    for (x, y), c in zip(data["p3d_flat_kpts"].astype(int), data["raw_colors_np"]):
        y_min = max(0, y - point_radius)
        y_max = min(flat_h, y + point_radius + 1)
        x_min = max(0, x - point_radius)
        x_max = min(flat_w, x + point_radius + 1)
        if y_min < y_max and x_min < x_max:
            flat_img[y_min:y_max, x_min:x_max] = c

    img_query_np = np.array(Image.open(data["img_query_path"]).convert("RGB")) / 255.0
    img_q_tensor = torch.from_numpy(img_query_np).float().permute(2, 0, 1)
    img_flat_tensor = torch.from_numpy(flat_img).float().permute(2, 0, 1)

    res_rbd = rbd(res)
    matches = res_rbd["matches"].cpu().numpy()
    
    m_kpts0 = data["q_kpts"][matches[..., 0]]
    m_kpts1 = data["p3d_flat_kpts"][matches[..., 1]]

    viz2d.plot_images([img_q_tensor, img_flat_tensor])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {res_rbd.get("stop", "all")} layers', fs=20)

    axes = plt.gcf().axes
    if len(axes) >= 2:
        h0, w0 = img_query_np.shape[:2]
        axes[0].set_xlim(0, w0)
        axes[0].set_ylim(h0, 0)
        axes[1].set_xlim(0, flat_w)
        axes[1].set_ylim(flat_h, 0)

    match_filename = f"flat_matches_scene_{scene}_{method}.png"
    plt.savefig(match_filename, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

    prune_filename = None
    if "prune0" in res_rbd:
        kpc0 = viz2d.cm_prune(res_rbd["prune0"])
        kpc1 = viz2d.cm_prune(res_rbd["prune1"])
        viz2d.plot_images([img_q_tensor, img_flat_tensor])
        viz2d.plot_keypoints([torch.from_numpy(data["q_kpts"]), torch.from_numpy(data["p3d_flat_kpts"])], colors=[kpc0, kpc1], ps=6)
        
        axes = plt.gcf().axes
        if len(axes) >= 2:
            axes[0].set_xlim(0, w0)
            axes[0].set_ylim(h0, 0)
            axes[1].set_xlim(0, flat_w)
            axes[1].set_ylim(flat_h, 0)
            
        prune_filename = f"flat_pruning_scene_{scene}_{method}.png"
        plt.savefig(prune_filename, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
    logger.info(f"Saved 2D Flat SfM images: {match_filename}" + (f" & {prune_filename}" if prune_filename else ""))

def main():
    parser = argparse.ArgumentParser(description="Visualize Matches in Rerun for MegaDepth")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to MegaDepth Undistorted_SfM root")
    parser.add_argument('--covisibility_dir', type=Path, required=True, help="Path to covisibility outputs")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query directory")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to SfM outputs")
    parser.add_argument('--depth_dir', type=Path, required=True, help="Path to depth maps")
    parser.add_argument('--scene', type=str, required=True, help="MegaDepth Scene name (e.g. 0015)")
    parser.add_argument('--query_name', type=str, default=None, help="Optional: Specific query image name to visualize")
    parser.add_argument('--method', type=str, required=True, choices=['MNN', 'PR', 'TRAIN'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--filter_threshold', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = args.method
    scene = args.scene

    logger.info(f"Starting Visualization for Scene: {scene} using method: {method}")

    # Initialize Loader
    loader = MegaDepthLoader(scene, args)
    if not loader.is_valid:
        return

    # Pick Query
    queries = loader.get_available_queries()
    if args.query_name:
        if args.query_name not in queries:
            logger.error(f"Provided query '{args.query_name}' not found.")
            return
        query_name = args.query_name
    else:
        query_name = random.choice(queries)

    # Get Data
    data = loader.get_query_data(query_name)
    if not data:
        logger.error("Failed to extract valid data for this query. Exiting.")
        return
        
    logger.info(f"Query: {data['query_name']} | Reference: {data['ref_name']}")

    # Initialize Matchers
    if method == "PR":
        logger.info("Initializing standard LightGlue for PR baseline evaluation...")
        baseline_matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    elif method == "TRAIN":
        if args.checkpoint is None: raise ValueError("--checkpoint must be provided when using the TRAIN method.")
        logger.info(f"Loaded trained LightGlu3D model with filter threshold {args.filter_threshold}...")
        lightglu3d_matcher = load_trained_lightglu3d(args.checkpoint, device, filter_threshold=args.filter_threshold)

    # Inference Execution
    res = None
    if method == "MNN":
        pred_matches0 = compute_nn_baseline(data["q_desc"], data["p3d_desc"], device)
    elif method == "PR":
        pred_matches0, res, data["p3d_flat_kpts"], flat_w, flat_h = compute_pr_baseline(
            baseline_matcher, data["q_kpts"], data["q_desc"], data["q_img_size"], 
            data["p3d_kpts"], data["p3d_desc"], data["ref_pose_matrix"], data["camera"], device
        )
    elif method == "TRAIN":
        pred_matches0, _ = compute_trained_lightglu3d(
            lightglu3d_matcher, data["q_kpts"], data["q_desc"], data["q_img_size"], 
            data["p3d_kpts"], data["p3d_desc"], device
        )

    # Evaluate metrics
    precision, recall, num_gt, num_pred, num_correct = compute_precision_recall(pred_matches0, data["gt_matches0"])
    
    logger.info("="*30)
    logger.info(f"{method} Results:")
    logger.info(f"GT Matches:        {num_gt}")
    logger.info(f"Predicted Matches: {num_pred}")
    logger.info(f"Correct Matches:   {num_correct}")
    logger.info(f"Precision:         {precision if precision is not None else 0.0:.4f}")
    logger.info(f"Recall:            {recall if recall is not None else 0.0:.4f}")
    logger.info("="*30)

    # Launch Visualizations
    if method == "PR":
        visual_flat_sfm(res, data, flat_w, flat_h, scene, method)

    launch_rerun_visualization(pred_matches0, data, scene, args, method_name=method)

if __name__ == "__main__":
    main()