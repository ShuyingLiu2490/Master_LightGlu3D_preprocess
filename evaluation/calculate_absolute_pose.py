import logging
import pycolmap
import numpy as np
from utils.utils import qvec2rotmat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_metrics(t_errors, r_errors, failed_pnp_count, total_queries, method_label):
    t_errs = np.array(t_errors)
    r_errs = np.array(r_errors)
    acc_strict = np.mean((t_errs <= 0.25) & (r_errs <= 2.0)) * 100
    acc_medium = np.mean((t_errs <= 0.50) & (r_errs <= 5.0)) * 100
    acc_loose  = np.mean((t_errs <= 5.00) & (r_errs <= 10.0)) * 100

    logger.info("="*40)
    logger.info(f"FINAL LOCALIZATION METRICS: {method_label}")
    logger.info("="*40)
    logger.info(f"Total Queries Evaluated:  {total_queries}")
    logger.info(f"Failed PnP Estimations:   {failed_pnp_count} images")
    logger.info(f"Median Trans Error:       {np.median(t_errs):.4f} m")
    logger.info(f"Median Rot Error:         {np.median(r_errs):.4f} deg")
    logger.info("-"*40)
    logger.info("Aachen Format (Translation & Rotation Limits):")
    logger.info(f"Strict (0.25m, 2°):     {acc_strict:.2f}%")
    logger.info(f"Medium (0.50m, 5°):     {acc_medium:.2f}%")
    logger.info(f"Loose  (5.00m, 10°):    {acc_loose:.2f}%")
    logger.info("="*40)

def evaluate_pose(matched_2d, matched_3d, camera, q_img_size, max_error): 
    orig_w, orig_h = camera["intrinsics"]["width"], camera["intrinsics"]["height"]
    new_w, new_h = q_img_size[0], q_img_size[1]
    params = np.array(camera["intrinsics"]["params"], dtype=float)
    
    if orig_w != new_w or orig_h != new_h:
        scale_x, scale_y = new_w / orig_w, new_h / orig_h
        model = camera["intrinsics"]["model"]
        if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
            params[0] *= scale_x; params[1] *= scale_x; params[2] *= scale_y
        elif model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"]:
            params[0] *= scale_x; params[1] *= scale_y; params[2] *= scale_x; params[3] *= scale_y

    colmap_cam = pycolmap.Camera(model=camera["intrinsics"]["model"], width=int(new_w), height=int(new_h), params=params)
    estimation_options = {"ransac": {"max_error": max_error}}
    refinement_options = {"refine_focal_length": False, "refine_extra_params": False}

    ret = pycolmap.estimate_and_refine_absolute_pose(matched_2d, matched_3d, colmap_cam, estimation_options, refinement_options)
    if ret is None or not ret.get("is_valid", True): return None

    if isinstance(ret, dict):
        R_est = ret["cam_from_world"].rotation.matrix()
        t_est = ret["cam_from_world"].translation
    else:
        R_est = qvec2rotmat(ret.qvec)
        t_est = ret.tvec

    R_gt = qvec2rotmat(camera["qvec"])
    t_gt = np.array(camera["tvec"]).reshape(3)

    C_gt = -R_gt.T @ t_gt
    C_est = -R_est.T @ t_est
    t_error = np.linalg.norm(C_est - C_gt)
    
    delta_R = R_est @ R_gt.T
    trace = np.clip(np.trace(delta_R), -1.0, 3.0)
    r_error_deg = np.degrees(np.arccos((trace - 1.0) / 2.0))

    return {"t_error": t_error, "r_error_deg": r_error_deg}

def estimate_pose_blind(matched_2d, matched_3d, camera, q_img_size, max_error): 
    orig_w, orig_h = camera.width, camera.height
    new_w, new_h = q_img_size[0], q_img_size[1]
    params = np.array(camera.params, dtype=float)
    
    if orig_w != new_w or orig_h != new_h:
        scale_x, scale_y = new_w / orig_w, new_h / orig_h
        model_name = getattr(camera, 'model_name', camera.model.name if hasattr(camera.model, 'name') else str(camera.model))
        if model_name in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL_FISHEYE", "SIMPLE_RADIAL_FISHEYE"]:
            params[0] *= scale_x; params[1] *= scale_x; params[2] *= scale_y 
        elif model_name in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "RADIAL"]:
            params[0] *= scale_x; params[1] *= scale_y; params[2] *= scale_x; params[3] *= scale_y 

    colmap_cam = pycolmap.Camera(model=camera.model, width=int(new_w), height=int(new_h), params=params)
    estimation_options = {"ransac": {"max_error": max_error}}
    refinement_options = {"refine_focal_length": False, "refine_extra_params": False}

    ret = pycolmap.estimate_and_refine_absolute_pose(matched_2d, matched_3d, colmap_cam, estimation_options, refinement_options)
    if ret is None or not ret.get("is_valid", True): return None

    if isinstance(ret, dict):
        qvec = ret["cam_from_world"].rotation.quat 
        tvec = ret["cam_from_world"].translation
    else:
        qvec = ret.qvec; tvec = ret.tvec

    return {"qvec": qvec, "tvec": tvec}