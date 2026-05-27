import argparse
from pathlib import Path
from hloc import extract_features, match_features, pairs_from_covisibility, triangulation

def build_sfm(args):
    dataset = args.dataset
    images = args.image_dir
    outputs = args.outputs
    
    sift_sfm = dataset / "3D-models/aachen_v_1_1"
    reference_sfm = outputs / "sfm_superpoint+lightglue"
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"

    # 2. Configurations for SuperPoint + LightGlue
    feature_conf = {
        'output': 'feats-superpoint-n2048',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 2048, 
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024, 
            "resize_force": False,
        },
    }
    
    matcher_conf = match_features.confs["superpoint+lightglue"]

    features = extract_features.main(feature_conf, images, outputs)

    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)
    
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    triangulation.main(
        reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Aachen SfM with SuperPoint + LightGlue")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the aachen_v1.1 dataset directory",)
    parser.add_argument("--image_dir", type=Path, required=True, help="Path to the images_upright directory",)
    parser.add_argument("--outputs", type=Path, required=True, help="Path to the output directory",)
    parser.add_argument("--num_covis", type=int, default=20, help="Number of overlapping image pairs for SfM triangulation, default: %(default)s",)
    args = parser.parse_args()
    
    build_sfm(args)