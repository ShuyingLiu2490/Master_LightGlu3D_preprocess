import torch
import numpy as np

def compute_hloc_baseline(baseline_matcher, q_kpts, q_desc, q_img_size, ref_data_list, p3d_indices_map, device):
    pred_matches0 = np.full(len(q_kpts), -1)
    for ref in ref_data_list:
        data = {
            "image0": {
                "keypoints": torch.from_numpy(q_kpts).unsqueeze(0).float().to(device),
                "descriptors": torch.from_numpy(q_desc.T).unsqueeze(0).float().to(device), 
                "image_size": torch.tensor([q_img_size]).float().to(device)
            },
            "image1": {
                "keypoints": torch.from_numpy(ref["kpts"]).unsqueeze(0).float().to(device),
                "descriptors": torch.from_numpy(ref["desc"].T).unsqueeze(0).float().to(device),
                "image_size": torch.tensor([ref["img_size"]]).float().to(device)
            }
        }
        with torch.no_grad():
            res = baseline_matcher(data)
        
        matches_2d = res['matches'][0].cpu().numpy() 
        ref_points3D_ids = ref["p3d_ids"]

        for q_idx, r_idx in matches_2d:
            if pred_matches0[q_idx] == -1:
                p3d_id = int(ref_points3D_ids[r_idx])
                if p3d_id != -1 and p3d_id != 18446744073709551615:
                    if p3d_id in p3d_indices_map:
                        pred_matches0[q_idx] = p3d_indices_map[p3d_id]
    return pred_matches0