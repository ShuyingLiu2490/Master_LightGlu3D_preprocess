import logging
import pickle
import numpy as np
import h5py
import pycolmap
from pathlib import Path
from PIL import Image
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat
from preprocess_Megadepth.generate_gt_pairs_by_scene import load_query_cams, compute_ground_truth_matches_soft
from utils.matching_performance import load_similar_pairs

logger = logging.getLogger(__name__)

class MegaDepthLoader:
    """Unified DataLoader for MegaDepth Inference and Visualization."""
    def __init__(self, scene, args):
        self.scene = scene
        self.args = args
        self.scene_query = args.query_dir / scene
        self.scene_covis = args.covisibility_dir / scene
        self.scene_sfm = args.sfm_dir / scene

        self.sfm_model_path = self.scene_sfm / "sfm_superpoint+lightglue"
        self.is_valid = self.sfm_model_path.exists()
        if not self.is_valid:
            logger.warning(f"SfM Model missing for {scene}.")
            return

        self.reconstruction = pycolmap.Reconstruction(self.sfm_model_path)
        _, self.images, _ = rw.read_model(self.sfm_model_path, ext=".bin")
        self.query_cams = load_query_cams(self.scene_query / "query_image_cameras.txt")
        self.pair_dict = load_similar_pairs(self.scene_covis / "most_similar_pairs.txt")

        with open(self.scene_covis / "covisibility_results.pkl", "rb") as f:
            self.covis_dict = pickle.load(f)

        query_names_file = self.scene_query / "query_image_names_clean.txt"
        with open(query_names_file, 'r') as f:
            self.queries = [line.strip() for line in f if line.strip()]

        self.q_feats_path = self.scene_sfm / "feats-superpoint-n2048.h5"
        self.p3d_feats_path = self.scene_covis / "points3D_feats_cache.h5"

    def get_available_queries(self):
        return self.queries

    def __len__(self):
        return len(self.queries)

    def __iter__(self):
        """Bulk Generator for Inference script."""
        with h5py.File(self.q_feats_path, "r") as q_feats_h5, h5py.File(self.p3d_feats_path, "r") as p3d_feats_h5:
            for query_name in self.queries:
                data = self._extract_data(query_name, q_feats_h5, p3d_feats_h5)
                if data:
                    yield data

    def get_query_data(self, query_name):
        """Single target loader for Visualization script."""
        with h5py.File(self.q_feats_path, "r") as q_feats_h5, h5py.File(self.p3d_feats_path, "r") as p3d_feats_h5:
            return self._extract_data(query_name, q_feats_h5, p3d_feats_h5)

    def _get_image_path(self, img_name):
        p1 = self.args.dataset / self.scene / "images" / img_name
        p2 = self.args.dataset / self.scene / img_name
        return p1 if p1.exists() else p2

    def _extract_data(self, query_name, q_feats_h5, p3d_feats_h5):
        ref_name = self.pair_dict.get(query_name)
        if not ref_name or query_name not in self.covis_dict or query_name not in q_feats_h5:
            return None

        visible_p3d = self.covis_dict[query_name]["unique_points"]
        if len(visible_p3d) == 0:
            return None

        # Load 2D Features
        q_kpts = q_feats_h5[query_name]["keypoints"][:]
        q_desc = q_feats_h5[query_name]["descriptors"][:]

        # Load 3D Features & Colors
        p3d_desc, p3d_kpts, raw_colors = [], [], []
        for pid in visible_p3d:
            pid_str, pid_int = str(pid), int(pid)
            if pid_str in p3d_feats_h5 and pid_int in self.reconstruction.points3D:
                p3d_desc.append(p3d_feats_h5[pid_str]["descriptors"][:].reshape(256))
                p3d_kpts.append(p3d_feats_h5[pid_str]["keypoints"][:].reshape(3))
                raw_colors.append(self.reconstruction.points3D[pid_int].color)

        if not p3d_kpts:
            return None

        p3d_desc = np.vstack(p3d_desc).T 
        p3d_kpts = np.vstack(p3d_kpts)   
        raw_pts_np = p3d_kpts.copy() 
        raw_colors_np = np.vstack(raw_colors) / 255.0

        # Camera & Image paths
        img_query_path = self._get_image_path(query_name)
        img_ref_path = self._get_image_path(ref_name)
        
        img_query_pil = Image.open(img_query_path)
        q_img_size = np.array([img_query_pil.width, img_query_pil.height])
        camera = self.query_cams[query_name]

        # Depth & Ground Truth
        depth_file = self.args.depth_dir / self.scene / f"{Path(query_name).stem}.h5"
        if not depth_file.exists():
            return None
            
        with h5py.File(depth_file, 'r') as f_depth:
            depth_map = f_depth['depth'][:]
                
        gt_matches0, _ = compute_ground_truth_matches_soft(
            {"keypoints": q_kpts}, {"keypoints": p3d_kpts}, camera, depth_map
        )

        # Ref Pose
        ref_image_obj = next((img for img in self.images.values() if img.name == ref_name), None)
        ref_pose_matrix = None
        if ref_image_obj is not None:
            ref_R = qvec2rotmat(ref_image_obj.qvec)
            ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))

        return {
            "query_name": query_name,
            "ref_name": ref_name,
            "img_query_path": img_query_path,
            "img_ref_path": img_ref_path,
            "q_kpts": q_kpts,
            "q_desc": q_desc,
            "q_img_size": q_img_size,
            "p3d_kpts": p3d_kpts,
            "p3d_desc": p3d_desc,
            "raw_pts_np": raw_pts_np,
            "raw_colors_np": raw_colors_np,
            "camera": camera,
            "ref_pose_matrix": ref_pose_matrix,
            "gt_matches0": gt_matches0
        }