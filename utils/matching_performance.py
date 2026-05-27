def load_similar_pairs(pair_file_path):
    pairs = {}
    if pair_file_path.exists():
        with open(pair_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pairs[parts[0]] = parts[1]
    return pairs

def compute_precision_recall(pred_matches, gt_matches):
    mask_recall = gt_matches > -1
    num_gt = mask_recall.sum()
    num_correct_recall = ((pred_matches == gt_matches) & mask_recall).sum()
    
    recall = (num_correct_recall / num_gt) if num_gt > 0 else None

    mask_precision = (pred_matches > -1) & (gt_matches >= -1)
    num_pred_eval = mask_precision.sum()
    num_correct_precision = ((pred_matches == gt_matches) & mask_precision).sum()
    
    precision = (num_correct_precision / num_pred_eval) if num_pred_eval > 0 else None

    return precision, recall, num_gt, num_pred_eval, num_correct_precision