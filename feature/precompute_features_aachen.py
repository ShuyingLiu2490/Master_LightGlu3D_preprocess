from pathlib import Path
from hloc.utils import read_write_model as rw
import argparse
import logging
from .precompute_features_re import extract_3d_descriptors, save_3d_features_to_h5

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_aachen(args):
    logger.info(f"Start averaged feature computation for Aachen...")
    
    # Define paths
    sfm_model_path = args.sfm_dir / "sfm_superpoint+lightglue"
    features_h5_path = args.sfm_dir / "feats-superpoint-n2048.h5"
    output_dir = args.outputs
    cached_feats_path = output_dir / "points3D_feats_cache.h5"
    
    if not sfm_model_path.exists():
        logger.error(f"SfM model not found at {sfm_model_path}")
        return
    if not features_h5_path.exists():
        logger.error(f"2D Features not found at {features_h5_path}")
        return
        
    if cached_feats_path.exists():
        logger.info(f"Skipping: Features already computed at {cached_feats_path}")
        return

    logger.info("Loading SfM model...")
    _, images, points3D = rw.read_model(sfm_model_path, ext=".bin")
    
    p3d_feats = extract_3d_descriptors(points3D, images, features_h5_path)
    
    logger.info(f"Extracted features for {len(p3d_feats)} 3D points in Aachen.")
    save_3d_features_to_h5(p3d_feats, cached_feats_path)
    logger.info(f"Averaged 3D features saved to {cached_feats_path}.")

def main():
    parser = argparse.ArgumentParser(description="Precompute 3D SuperPoint Features for Aachen")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Aachen root")
    parser.add_argument('--outputs', type=Path, required=True, help="Path to output directory")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to the directory containing sfm models")
    
    args = parser.parse_args()
    
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset}")

    try:
        process_aachen(args)
    except Exception as e:
        logger.error(f"Failed to process Aachen features: {e}", exc_info=True)

if __name__ == "__main__":
    main()