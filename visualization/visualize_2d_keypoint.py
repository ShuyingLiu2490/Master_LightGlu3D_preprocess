import argparse
import logging
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import cv2
from lightglue import SuperPoint
from lightglue.utils import load_image

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_confidence(image_path, keypoints, scores, output_filename, threshold=0.015, max_points=2048):
    # Load image (converting Path object to string for OpenCV)
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Split points based on the natural threshold
    natural_mask = scores >= threshold
    forced_mask = scores < threshold
    
    natural_kpts = keypoints[natural_mask]
    natural_scores = scores[natural_mask]
    
    forced_kpts = keypoints[forced_mask]
    forced_scores = scores[forced_mask]
    
    # Calculate averages
    avg_natural = np.mean(natural_scores) if len(natural_scores) > 0 else 0
    avg_forced = np.mean(forced_scores) if len(forced_scores) > 0 else 0
    avg_total = np.mean(scores)
    
    # Print the info using the logger
    logger.info("========================================")
    logger.info("Superpoint Feature Confidence Analysis")
    logger.info(f"Total Points Extracted: {len(scores)} (Target Max: {max_points})")
    logger.info(f"Natural Points (>= {threshold}): {len(natural_scores)} points | Avg Score: {avg_natural:.4f}")
    logger.info(f"Forced Points  (< {threshold}): {len(forced_scores)} points | Avg Score: {avg_forced:.4f}")
    logger.info(f"Overall Average Score:  {avg_total:.4f}")
    logger.info("========================================")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_rgb)

    log_vmin = 1e-4 # Minimum expected score
    log_vmax = np.max(scores)
    log_norm = LogNorm(vmin=log_vmin, vmax=log_vmax)
    
    # Plot natural points mapped to a colorbar (only if there are any)
    if len(natural_scores) > 0:
        sc_natural = ax.scatter(natural_kpts[:, 0], natural_kpts[:, 1],
                                c=natural_scores, cmap='viridis', 
                                marker='o', s=15, norm=log_norm,
                                label=f'Natural (Score >= {threshold})')
        # Add the colorbar to the side
        cbar = plt.colorbar(sc_natural, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SuperPoint Confidence Score', rotation=270, labelpad=15)
    
    # Plot forced points (red x) to highlight (only if there are any)
    if len(forced_scores) > 0:
        sc_forced = ax.scatter(forced_kpts[:, 0], forced_kpts[:, 1],
                               color='red', marker='x', s=20, alpha=0.7,
                               label=f'Forced (Score < {threshold})')
    
    # Add titles and legend
    plt.title(f"Keypoint Confidence\nNatural Avg: {avg_natural:.4f} | Forced Avg: {avg_forced:.4f}", fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 0.0), 
              borderaxespad=0., title='Keypoint Types', fontsize=10)
    plt.axis('off')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved 2D keypoint visualization to {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Extract and Visualize SuperPoint 2D Keypoints")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to Undistorted_SfM")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query sets")
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--max_kpts', type=int, default=2048, help="Max keypoints to extract")
    parser.add_argument('--threshold', type=float, default=0.001, help="Score threshold distinguishing natural vs forced points")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = args.scene
    logger.info(f"Starting 2D keypoint analysis for Scene {scene}...")

    # Randomly pick an image
    cam_file = args.query_dir / scene / "query_image_cameras.txt"
    with open(cam_file, 'r') as f:
        image_names = [line.strip().split()[0] for line in f if line.strip() and not line.startswith('#')]
    
    query_name = random.choice(image_names)
    image_path = args.dataset / scene / "images" / query_name
    logger.info(f"Selected image: {query_name}")

    # Load image
    image_tensor = load_image(image_path)
    
    # Initialize SuperPoint
    logger.info("Initializing SuperPoint extractor...")
    extractor = SuperPoint(max_num_keypoints=args.max_kpts).eval().to(device)

    # Extract features
    with torch.no_grad():
        feats = extractor.extract(image_tensor.unsqueeze(0).to(device))
    
    # Unpack the batch
    kpts = feats['keypoints'][0].cpu().numpy()
    scores = feats['keypoint_scores'][0].cpu().numpy()

    logger.info("Generating visualization...")
    
    # Define output filename based on the query image name
    output_filename = f"confidence_kpts_{Path(query_name).stem}.png"
    
    # Run the new confidence visualization
    visualize_confidence(
        image_path=image_path, 
        keypoints=kpts, 
        scores=scores, 
        output_filename=output_filename,
        threshold=args.threshold,
        max_points=args.max_kpts
    )

if __name__ == "__main__":
    main()