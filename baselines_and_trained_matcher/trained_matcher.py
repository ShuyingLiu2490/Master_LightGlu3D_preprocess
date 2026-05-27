# export PYTHONPATH="/home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match:$PYTHONPATH"

import numpy as np
import torch
import logging
from gluefactory.models.matchers.lightglu3d_bicross import LightGlu3D

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add filter_threshold as parameter, usually set to 0.1, 0.05, 0.025, 0.015
def load_trained_lightglu3d(checkpoint_path, device, filter_threshold=0.1):
    logger.info(f"Loading trained LightGlu3D from {checkpoint_path}...")
    conf = {
        "name": "lightglu3d_bicross", 
        "input_dim": 256, 
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,
        "mp": False, 
        "depth_confidence": -1, 
        "width_confidence": -1, 
        "filter_threshold": filter_threshold,
        "checkpointed": False,
    }
    matcher = LightGlu3D(conf).eval().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) 
    matcher.load_state_dict(state_dict, strict=False)
    return matcher

def compute_trained_lightglu3d(matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device):
    data = {
        "keypoints0": torch.from_numpy(q_kpts).unsqueeze(0).float().to(device),
        "keypoints1": torch.from_numpy(p3d_kpts).unsqueeze(0).float().to(device),
        "descriptors0": torch.from_numpy(q_desc.T).unsqueeze(0).float().to(device), 
        "descriptors1": torch.from_numpy(p3d_desc.T).unsqueeze(0).float().to(device), 
        "view0": {
            "image_size": torch.from_numpy(q_img_size).unsqueeze(0).float().to(device)
        }
    }
    with torch.no_grad():
        pred = matcher(data)
        log_assign_matrix = pred["log_assignment"][0].detach().cpu().numpy()
    
    return pred['matches0'][0].cpu().numpy(), log_assign_matrix

def filter_matches_greedy(scores: torch.Tensor, th: float):
    """
    One-to-one greedy matching without mutual NN.

    Args:
        scores: [M+1, N+1] log assignment matrix
        th: threshold on exp(score)

    Returns:
        m0: [M] row -> col
        m1: [N] col -> row
        mscores0: [M]
        mscores1: [N]
    """
    scores = scores[:-1, :-1]
    M, N = scores.shape
    m0_raw = np.argmax(scores, axis=1)
    max_scores = scores[np.arange(M), m0_raw]
    mscores0_raw = np.exp(max_scores)

    m0 = -np.ones(M, dtype=np.int32)
    m1 = -np.ones(N, dtype=np.int32)
    mscores0 = np.zeros(M, dtype=np.float32)
    mscores1 = np.zeros(N, dtype=np.float32)
    valid = mscores0_raw > th

    rows = np.arange(M)[valid]
    cols = m0_raw[valid]
    vals = mscores0_raw[valid]

    # sort descending by score
    order = np.argsort(-vals)
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]

    used_rows = set()
    used_cols = set()
    # greedy one-to-one assignment
    for r, c, v in zip(rows, cols, vals):
        if r in used_rows: continue
        if c in used_cols: continue
        m0[r] = c
        m1[c] = r
        mscores0[r] = v
        mscores1[c] = v
        used_rows.add(r)
        used_cols.add(c)

    return m0, m1, mscores0, mscores1

# def compute_trained_lightglu3d_greedy_dynamic(matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device, threshold_stats, min_matches=100):
def compute_trained_lightglu3d_greedy_dynamic(matcher, q_kpts, q_desc, q_img_size, p3d_kpts, p3d_desc, device, min_matches=100):   
    """Runs the forward pass and dynamically adjusts the score threshold to ensure at least `min_matches` are returned if possible."""
    data = {
        "keypoints0": torch.from_numpy(q_kpts).unsqueeze(0).float().to(device),
        "keypoints1": torch.from_numpy(p3d_kpts).unsqueeze(0).float().to(device),
        "descriptors0": torch.from_numpy(q_desc.T).unsqueeze(0).float().to(device), 
        "descriptors1": torch.from_numpy(p3d_desc.T).unsqueeze(0).float().to(device), 
        "view0": {
            "image_size": torch.from_numpy(q_img_size).unsqueeze(0).float().to(device)
        }
    }
    with torch.no_grad():
        pred = matcher(data)
        log_assign_matrix = pred["log_assignment"][0].detach().cpu().numpy()
        
    # if min_matches <= 400:
        # thresholds_to_try = [0.05, 0.025, 0.015, 0.005] # for 100, 250, 400
    # else: 
        # thresholds_to_try = [0.05, 0.025, 0.015, 0.005, 0.001] # for 800, 1000
    thresholds_to_try = [0.05, 0.025, 0.015, 0.005, 0.001]
    
    for th in thresholds_to_try:
        m0, _, _, _ = filter_matches_greedy(log_assign_matrix, th=th)
        valid_matches = np.sum(m0 > -1)
        if valid_matches >= min_matches:
            # threshold_stats[th] += 1
            break
    # if valid_matches < min_matches:
        # threshold_stats[thresholds_to_try[-1]] += 1
            
    return m0, log_assign_matrix