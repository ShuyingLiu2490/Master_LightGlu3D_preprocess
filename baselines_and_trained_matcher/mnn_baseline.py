# export PYTHONPATH="/home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match:$PYTHONPATH"

import torch
from gluefactory.models import get_model

def compute_nn_baseline(q_desc, p3d_desc, device):
    nn_conf = {
        "name": "matchers.nearest_neighbor_matcher",
        "mutual_check": True,
        "ratio_thresh": None,
        "distance_thresh": 0.75,
    }
    matcher = get_model(nn_conf["name"])(nn_conf).eval().to(device)

    data = {
        "descriptors0": torch.from_numpy(q_desc.T).unsqueeze(0).float().to(device),
        "descriptors1": torch.from_numpy(p3d_desc.T).unsqueeze(0).float().to(device)
    }

    with torch.no_grad():
        pred = matcher(data)
    
    pred_matches0 = pred['matches0'][0].cpu().numpy() 
    return pred_matches0