import pickle
import numpy as np
import h5py
from PIL import Image
import pycolmap
from hloc.utils import read_write_model as rw
from utils.utils import qvec2rotmat, get_most_similar_ref

class CambridgeLoader:
    def __init__(self, scene, args):
        self.scene = scene
        self.args = args
        self.sfm_model_path = args.sfm_dir / scene / "sfm_superpoint+lightglue"
        
        self.is_valid = self.sfm_model_path.exists()
        if not self.is_valid: return

        self.reconstruction = pycolmap.Reconstruction(self.sfm_model_path)
        _, self.images, _ = rw.read_model(self.sfm_model_path, ext=".bin")
        
        with open(args.covisibility_dir / scene / "covisibility_results.pkl", "rb") as f:
            self.covis_dict = pickle.load(f)

        gt_model_path = args.query_dir / scene / "empty_all"
        cameras_gt, images_gt, _ = rw.read_model(gt_model_path, ext=".txt")
        self.query_cams = {}
        for img in images_gt.values():
            cam = cameras_gt[img.camera_id]
            self.query_cams[img.name] = {
                "qvec": img.qvec, "tvec": img.tvec,
                "intrinsics": {"model": cam.model, "width": cam.width, "height": cam.height, "params": cam.params}
            }
        
        with open(args.query_dir / scene / "list_query.txt", 'r') as f:
            self.queries = [line.strip() for line in f if line.strip()]

        self.active_pair_file = args.covisibility_dir / scene / "most_similar_pairs.txt"
        self.img_dir_base = args.dataset / scene
        self.q_feats_path = args.sfm_dir / scene / "feats-superpoint-n2048.h5"
        self.p3d_feats_path = args.covisibility_dir / scene / "points3D_feats_cache.h5"

    def __len__(self):
        return len(self.queries)

    def __iter__(self):
        with h5py.File(self.q_feats_path, "r") as q_feats_h5, h5py.File(self.p3d_feats_path, "r") as p3d_feats_h5:
            for query_name in self.queries:
                if query_name not in self.covis_dict or query_name not in self.query_cams: continue
                
                primary_ref = get_most_similar_ref(query_name, self.active_pair_file)
                if not primary_ref or query_name not in q_feats_h5: 
                    yield {"query_name": query_name, "is_valid": False}; continue

                # Load 2D
                q_kpts = q_feats_h5[query_name]["keypoints"][:]
                q_desc = q_feats_h5[query_name]["descriptors"][:]
                
                img_query_pil = Image.open(self.img_dir_base / query_name)
                q_img_size = np.array([img_query_pil.width, img_query_pil.height])

                # Load 3D
                visible_p3d = self.covis_dict[query_name]["unique_points"]
                p3d_desc, p3d_kpts = [], []
                p3d_indices_map = {}
                idx_counter = 0

                for pid in visible_p3d:
                    pid_str, pid_int = str(pid), int(pid)
                    if pid_str in p3d_feats_h5 and pid_int in self.reconstruction.points3D:
                        p3d_desc.append(p3d_feats_h5[pid_str]["descriptors"][:].reshape(256))
                        p3d_kpts.append(p3d_feats_h5[pid_str]["keypoints"][:].reshape(3))
                        p3d_indices_map[pid_int] = idx_counter
                        idx_counter += 1

                if not p3d_kpts:
                    yield {"query_name": query_name, "is_valid": False}; continue

                # Ref Pose
                ref_image_obj = next((img for img in self.images.values() if img.name == primary_ref), None)
                ref_R = qvec2rotmat(ref_image_obj.qvec)
                ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))

                # HLOC top refs
                ref_data_list = []
                if self.args.method == "HLOC":
                    valid_image_ids = self.covis_dict[query_name].get('unique_images', set())
                    top_refs = [self.images[img_id].name for img_id in valid_image_ids if img_id in self.images]
                    if primary_ref in top_refs: top_refs.remove(primary_ref)
                    top_refs.insert(0, primary_ref)

                    for r_name in top_refs:
                        if r_name in q_feats_h5:
                            r_img_pil = Image.open(self.img_dir_base / r_name)
                            r_img_obj = next((img for img in self.images.values() if img.name == r_name), None)
                            if r_img_obj:
                                ref_data_list.append({
                                    "kpts": q_feats_h5[r_name]["keypoints"][:], "desc": q_feats_h5[r_name]["descriptors"][:],
                                    "img_size": [r_img_pil.width, r_img_pil.height],
                                    "p3d_ids": r_img_obj.point3D_ids, "name": r_name, "image_id": r_img_obj.id
                                })

                camera_dict = self.query_cams[query_name]
                colmap_cam = pycolmap.Camera(
                    model=camera_dict["intrinsics"]["model"], width=int(camera_dict["intrinsics"]["width"]),
                    height=int(camera_dict["intrinsics"]["height"]), params=np.array(camera_dict["intrinsics"]["params"], dtype=float)
                )

                yield {
                    "is_valid": True, "query_name": query_name, "q_kpts": q_kpts, "q_desc": q_desc, "q_img_size": q_img_size,
                    "p3d_kpts": np.vstack(p3d_kpts), "p3d_desc": np.vstack(p3d_desc).T, "p3d_for_pnp": np.vstack(p3d_kpts),
                    "p3d_indices_map": p3d_indices_map, "ref_pose_matrix": ref_pose_matrix, "camera_dict": camera_dict,
                    "colmap_cam": colmap_cam, "ref_data_list": ref_data_list
                }


class AachenLoader:
    def __init__(self, args):
        self.args = args
        self.sfm_model_path = args.sfm_dir / "sfm_superpoint+lightglue"
        self.reconstruction = pycolmap.Reconstruction(self.sfm_model_path)
        _, self.images, _ = rw.read_model(self.sfm_model_path, ext=".bin")
        
        with open(args.covisibility_dir / "covisibility_results.pkl", "rb") as f:
            self.covis_dict = pickle.load(f)
            
        self.aachen_query_cams = {}
        for query_file in [args.query_dir / "day_time_queries_with_intrinsics.txt", args.query_dir / "night_time_queries_with_intrinsics.txt"]:
            if query_file.exists():
                with open(query_file, 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith("#"):
                            parts = line.strip().split()
                            self.aachen_query_cams[parts[0]] = pycolmap.Camera(
                                model=parts[1], width=int(parts[2]), height=int(parts[3]), params=np.array(parts[4:], dtype=float)
                            )
        
        with open(args.covisibility_dir / "clean_aachen_queries.txt", 'r') as f:
            self.queries = [line.strip() for line in f if line.strip()]

        self.active_pair_file = args.covisibility_dir / "most_similar_pairs.txt"
        self.q_feats_path = args.sfm_dir / "feats-superpoint-n2048.h5"
        self.p3d_feats_path = args.covisibility_dir / "points3D_feats_cache.h5"

    def __len__(self):
        return len(self.queries)

    def __iter__(self):
        with h5py.File(self.q_feats_path, "r") as q_feats_h5, h5py.File(self.p3d_feats_path, "r") as p3d_feats_h5:
            for query_name in self.queries:
                if query_name not in self.covis_dict or query_name not in self.aachen_query_cams: continue

                primary_ref = get_most_similar_ref(query_name, self.active_pair_file)
                if not primary_ref or query_name not in q_feats_h5: 
                    yield {"query_name": query_name, "is_valid": False}; continue

                # Load 2D
                q_kpts = q_feats_h5[query_name]["keypoints"][:]
                q_desc = q_feats_h5[query_name]["descriptors"][:]
                q_img_size = np.array(q_feats_h5[query_name]["image_size"][:])

                # Load 3D
                visible_p3d = self.covis_dict[query_name]["unique_points"]
                p3d_desc, p3d_kpts, p3d_xyz = [], [], []
                p3d_indices_map = {}
                idx_counter = 0

                for pid in visible_p3d:
                    pid_str, pid_int = str(pid), int(pid)
                    if pid_str in p3d_feats_h5 and pid_int in self.reconstruction.points3D:
                        p3d_desc.append(p3d_feats_h5[pid_str]["descriptors"][:].reshape(256))
                        p3d_kpts.append(p3d_feats_h5[pid_str]["keypoints"][:].reshape(3))
                        p3d_xyz.append(self.reconstruction.points3D[pid_int].xyz)
                        p3d_indices_map[pid_int] = idx_counter
                        idx_counter += 1

                if not p3d_kpts:
                    yield {"query_name": query_name, "is_valid": False}; continue

                # Ref Pose
                ref_image_obj = next((img for img in self.images.values() if img.name == primary_ref), None)
                ref_R = qvec2rotmat(ref_image_obj.qvec)
                ref_pose_matrix = np.hstack((ref_R, ref_image_obj.tvec.reshape(3, 1)))

                # HLOC top refs
                ref_data_list = []
                if self.args.method == "HLOC":
                    valid_image_ids = self.covis_dict[query_name].get('unique_images', set())
                    top_refs = [self.images[img_id].name for img_id in valid_image_ids if img_id in self.images]
                    if primary_ref in top_refs: top_refs.remove(primary_ref)
                    top_refs.insert(0, primary_ref)

                    for r_name in top_refs:
                        if r_name in q_feats_h5:
                            r_img_obj = next((img for img in self.images.values() if img.name == r_name), None)
                            if r_img_obj:
                                ref_data_list.append({
                                    "kpts": q_feats_h5[r_name]["keypoints"][:], "desc": q_feats_h5[r_name]["descriptors"][:],
                                    "img_size": q_feats_h5[r_name]["image_size"][:],
                                    "p3d_ids": r_img_obj.point3D_ids, "name": r_name, "image_id": r_img_obj.id
                                })

                colmap_cam = self.aachen_query_cams[query_name]
                camera_dict = {
                    "intrinsics": {
                        "model": getattr(colmap_cam, 'model_name', getattr(colmap_cam.model, 'name', str(colmap_cam.model))),
                        "width": colmap_cam.width, "height": colmap_cam.height, "params": colmap_cam.params
                    }
                }

                yield {
                    "is_valid": True, "query_name": query_name, "q_kpts": q_kpts, "q_desc": q_desc, "q_img_size": q_img_size,
                    "p3d_kpts": np.vstack(p3d_kpts), "p3d_desc": np.vstack(p3d_desc).T, "p3d_for_pnp": np.vstack(p3d_xyz), # Aachen uses XYZ for PnP
                    "p3d_indices_map": p3d_indices_map, "ref_pose_matrix": ref_pose_matrix, "camera_dict": camera_dict,
                    "colmap_cam": colmap_cam, "ref_data_list": ref_data_list
                }