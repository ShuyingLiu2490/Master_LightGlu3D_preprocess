# From the argument get the dataset path, preprocess path, scene number
# Process (for each scene):
# 1. get one query image (random)
#    then get the most similar reference image (from most_similar_pair.txt)
# 2. get the original sfm model
#    then get the visible 3d points for this query image (from covisibility_results.pkl)
# 3. use the normalization methods (provided) to deal with the visible 3d points
#    then visualize it in html file

import argparse
import logging
import pickle
import random
import numpy as np
import torch
import pycolmap
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hloc.utils import viz_3d

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_keypoints(
    kpts: torch.Tensor
) -> torch.Tensor:
    size = kpts.max(-2).values
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)

    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]

    return kpts

# Normalization function for 3d points from ligthglu3d matchers
def normalize_3d_with_quantile(
    kpts: torch.Tensor, quantile_value:float=0.975
) -> torch.Tensor:
    upper_bound = torch.quantile(kpts, quantile_value, dim=-2, keepdim=True)
    lower_bound = torch.quantile(kpts, 1 - quantile_value, dim=-2, keepdim=True)
    
    shift = (upper_bound + lower_bound) / 2
    dist = (upper_bound - lower_bound)
    scale = dist.max(dim=-1, keepdim=True).values / 2.0
    scale = torch.clamp(scale, min=1e-6)

    # 1. Base normalization
    kpts_norm_base = (kpts - shift) / scale
    norm_upper_base = (upper_bound - shift) / scale
    norm_lower_base = (lower_bound - shift) / scale

    # 2. Pull back
    pull_factor = (2 * quantile_value - 1)
    kpts_norm_pulled = kpts_norm_base * pull_factor
    norm_upper_pulled = norm_upper_base * pull_factor
    norm_lower_pulled = norm_lower_base * pull_factor

    return (kpts_norm_pulled, kpts_norm_base, 
            upper_bound, lower_bound, 
            norm_upper_pulled, norm_lower_pulled)

def get_most_similar_ref(query_name: str, pair_file_path: Path):
    if not pair_file_path.exists():
        return None
    with open(pair_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == query_name:
                return parts[1]
    return None

def create_box_frame(min_b, max_b):
    x0, y0, z0 = min_b
    x1, y1, z1 = max_b
    
    # Path connecting the 12 edges of a 3D box
    x = [x0, x1, x1, x0, x0, x0, x1, x1, x0, x0, None, x1, x1, None, x1, x1, None, x0, x0]
    y = [y0, y0, y1, y1, y0, y0, y0, y1, y1, y0, None, y0, y0, None, y1, y1, None, y1, y1]
    z = [z0, z0, z0, z0, z0, z1, z1, z1, z1, z1, None, z0, z1, None, z0, z1, None, z0, z1]
    
    return x, y, z

def format_vec(vec):
    return f"X={vec[0]:.2f}, Y={vec[1]:.2f}, Z={vec[2]:.2f}"

def main():
    parser = argparse.ArgumentParser(description="Visualize 3D Point Cloud Normalization with 3 Views")
    parser.add_argument('--covisibility_dir', type=Path, required=True, help="Path to covisibility")
    parser.add_argument('--query_dir', type=Path, required=True, help="Path to query_sets")
    parser.add_argument('--sfm_dir', type=Path, required=True, help="Path to triangulation outputs")
    parser.add_argument('--scene', type=str, required=True, help="Scene ID")
    parser.add_argument('--quantile', type=float, default=0.975, help="Quantile value for normalization")
    args = parser.parse_args()

    scene = args.scene
    logger.info(f"Visualizing One Random Query Normalization for Scene {scene}...")

    query_names_file = args.query_dir / scene / "query_image_names.txt"
    pair_file = args.covisibility_dir / scene / "most_similar_pairs.txt"
    covis_file = args.covisibility_dir / scene / "covisibility_results.pkl"
    sfm_model_path = args.sfm_dir / scene / "sfm_superpoint+lightglue"
    
    # Select a random query
    with open(query_names_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    query_name = random.choice(queries)
    ref_name = get_most_similar_ref(query_name, pair_file)
    logger.info(f"Selected Query: {query_name} (Ref: {ref_name})")

    # Get visible 3D points
    with open(covis_file, "rb") as f:
        covis_dict = pickle.load(f)
    if query_name not in covis_dict:
        logger.error(f"Covisibility data missing for {query_name}.")
        return
    visible_p3d_ids = covis_dict[query_name]["unique_points"]

    # Load original sfm model
    reconstruction = pycolmap.Reconstruction(sfm_model_path)
    raw_coords, raw_colors = [], []
    for pid in visible_p3d_ids:
        pid = int(pid)
        if pid in reconstruction.points3D:
            pt = reconstruction.points3D[pid]
            raw_coords.append(pt.xyz)
            raw_colors.append(pt.color)
    if not raw_coords:
        logger.error("No valid 3D coordinates found in the reconstruction.")
        return
    raw_pts_np = np.vstack(raw_coords)
    raw_colors_np = np.vstack(raw_colors)

    # Apply normalization
    logger.info("Applying Quantile Normalization...")
    raw_pts_tensor = torch.from_numpy(raw_pts_np).float()
    quantile_value = args.quantile
    pull_factor = (2 * quantile_value - 1)
    
    # Fetch all returned values
    (norm_pts_pulled_t, norm_pts_base_t, 
     raw_upper_t, raw_lower_t, 
     norm_upper_pulled_t, norm_lower_pulled_t) = normalize_3d_with_quantile(raw_pts_tensor, quantile_value)
    norm_pts_pulled = norm_pts_pulled_t.numpy()
    norm_pts_base = norm_pts_base_t.numpy()
    raw_upper = raw_upper_t.squeeze().numpy()
    raw_lower = raw_lower_t.squeeze().numpy()
    norm_upper_pulled = norm_upper_pulled_t.squeeze().numpy()
    norm_lower_pulled = norm_lower_pulled_t.squeeze().numpy()
    
    # Calculate region and size
    raw_min, raw_max = raw_pts_np.min(axis=0), raw_pts_np.max(axis=0)
    base_min, base_max = norm_pts_base.min(axis=0), norm_pts_base.max(axis=0)
    pulled_min, pulled_max = norm_pts_pulled.min(axis=0), norm_pts_pulled.max(axis=0)
    raw_size_whole = raw_max - raw_min
    raw_size_core = raw_upper - raw_lower
    base_size_whole = base_max - base_min
    pulled_size_whole = pulled_max - pulled_min
    pulled_size_core = norm_upper_pulled - norm_lower_pulled

    logger.info("=" * 60)
    logger.info("1. Raw 3d points: ")
    logger.info(f"   Whole region : Min[{format_vec(raw_min)}]  Max[{format_vec(raw_max)}]")
    logger.info(f"   Whole size   : {format_vec(raw_size_whole)}")
    logger.info(f"   Core bound   : Min[{format_vec(raw_lower)}]  Max[{format_vec(raw_upper)}]")
    logger.info(f"   Core size    : {format_vec(raw_size_core)}")
    logger.info("-" * 60)
    
    logger.info("2. Base normalization: ")
    logger.info(f"   Whole region : Min[{format_vec(base_min)}]  Max[{format_vec(base_max)}]")
    logger.info(f"   Whole size   : {format_vec(base_size_whole)}")
    logger.info("-" * 60)
    
    logger.info("3. Pull back: ")
    logger.info(f"   Whole region : Min[{format_vec(pulled_min)}]  Max[{format_vec(pulled_max)}]")
    logger.info(f"   Whole size   : {format_vec(pulled_size_whole)}")
    logger.info("=" * 60)

    # Calculate points inside valid [-1, 1]
    total_pts = len(norm_pts_base)
    inside_base = np.sum(np.all(np.abs(norm_pts_base) <= 1.0, axis=1))
    inside_pulled = np.sum(np.all(np.abs(norm_pts_pulled) <= 1.0, axis=1))

    logger.info("=" * 60)
    logger.info("Inside valid cube [-1,1]: ")
    logger.info(f"Total visible points      : {total_pts}")
    logger.info(f"Points inside (base)      : {inside_base} ({inside_base/total_pts*100:.1f}%)")
    logger.info(f"Points inside (pull back) : {inside_pulled} ({inside_pulled/total_pts*100:.1f}%)")
    logger.info(f"Points rescued            : +{inside_pulled - inside_base}")
    logger.info("=" * 60 + "\n")

    # Create html file with 3 views
    fig = make_subplots(
        rows=1, cols=3, 
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            "Original SfM + Visible (Red)", 
            f"Raw Visible Points<br>Core Size: {raw_size_core[0]:.1f} x {raw_size_core[1]:.1f} x {raw_size_core[2]:.1f}",
            f"Normalized Points<br>Core Size: {pulled_size_core[0]:.2f} x {pulled_size_core[1]:.2f} x {pulled_size_core[2]:.2f}"
        )
    )

    # Left: original + red visible
    viz_3d.plot_reconstruction(
        fig, reconstruction, color='rgba(200,200,200,0.1)', name="Full SfM Geometry"
    )
    fig.add_trace(go.Scatter3d(
        x=raw_pts_np[:, 0], y=raw_pts_np[:, 1], z=raw_pts_np[:, 2],
        mode='markers',
        marker=dict(size=2, color='red', opacity=0.8),
        name="Visible Points"
    ), row=1, col=1)

    # Middle: visible + yellow core box
    marker_colors = [f'rgb({c[0]},{c[1]},{c[2]})' for c in raw_colors_np]
    fig.add_trace(go.Scatter3d(
        x=raw_pts_np[:, 0], y=raw_pts_np[:, 1], z=raw_pts_np[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=marker_colors, opacity=1.0),
        name="Raw Visible Points"
    ), row=1, col=2)
    
    bx_raw, by_raw, bz_raw = create_box_frame(raw_lower, raw_upper)
    fig.add_trace(go.Scatter3d(
        x=bx_raw, y=by_raw, z=bz_raw, mode='lines',
        line=dict(color='yellow', width=3), name="Majority Bound"
    ), row=1, col=2)

    # Right: normalized + green valid box + blue base valid box
    fig.add_trace(go.Scatter3d(
        x=norm_pts_pulled[:, 0], y=norm_pts_pulled[:, 1], z=norm_pts_pulled[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=marker_colors, opacity=1.0),
        name="Normalized Visible"
    ), row=1, col=3)

    bx_valid, by_valid, bz_valid = create_box_frame([-pull_factor, -pull_factor, -pull_factor], [pull_factor, pull_factor, pull_factor])
    fig.add_trace(go.Scatter3d(
        x=bx_valid, y=by_valid, z=bz_valid, mode='lines',
        line=dict(color='blue', width=1), name="Base Valid Region"
    ), row=1, col=3)

    bx_valid, by_valid, bz_valid = create_box_frame([-1, -1, -1], [1, 1, 1])
    fig.add_trace(go.Scatter3d(
        x=bx_valid, y=by_valid, z=bz_valid, mode='lines',
        line=dict(color='green', width=1), name="New Valid Region"
    ), row=1, col=3)

    fig.update_layout(
        title=f"2D-3D Covisibility & Normalization: scene {scene}: {query_name}",
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data'),
        scene3=dict(aspectmode='data'),
        height=700,
        width=1800,
        showlegend=True,
        template="plotly_dark" 
    )

    output_html = Path(f"normalization_check_3views_{scene}_2.html")
    fig.write_html(str(output_html))
    logger.info(f"HTML Visualization successfully saved to {output_html}")

    # Original visualization
    # Save in a new html
    # Left is the original normalization, right is the updated normalization
    logger.info("Generating Side-by-Side Normalization Comparison...")

    orig_norm_pts_t = normalize_keypoints(raw_pts_tensor)
    orig_norm_pts = orig_norm_pts_t.numpy()

    # Calculate points inside valid [-1, 1] for original method
    inside_orig = np.sum(np.all(np.abs(orig_norm_pts) <= 1.0, axis=1))
    
    logger.info("=" * 60)
    logger.info("Original Normalization Stats: ")
    logger.info(f"Points inside [-1, 1]     : {inside_orig} ({inside_orig/total_pts*100:.1f}%)")
    logger.info("=" * 60 + "\n")

    # Create html file with 2 views
    fig_compare = make_subplots(
        rows=1, cols=2, 
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            f"Original Normalization<br>Inside [-1,1]: {inside_orig/total_pts*100:.1f}%", 
            f"Quantile Normalization + Pull<br>Inside [-1,1]: {inside_pulled/total_pts*100:.1f}%"
        )
    )

    # Left: Original Normalization
    fig_compare.add_trace(go.Scatter3d(
        x=orig_norm_pts[:, 0], y=orig_norm_pts[:, 1], z=orig_norm_pts[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=marker_colors, opacity=1.0),
        name="Orig Normalized Points"
    ), row=1, col=1)

    # Standard [-1, 1] Box for reference
    bx_valid, by_valid, bz_valid = create_box_frame([-1, -1, -1], [1, 1, 1])
    fig_compare.add_trace(go.Scatter3d(
        x=bx_valid, y=by_valid, z=bz_valid, mode='lines',
        line=dict(color='green', width=1), name="Target [-1, 1] Region"
    ), row=1, col=1)

    # Right: Updated Normalization (Same as the right-most view from before)
    fig_compare.add_trace(go.Scatter3d(
        x=norm_pts_pulled[:, 0], y=norm_pts_pulled[:, 1], z=norm_pts_pulled[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=marker_colors, opacity=1.0),
        name="New Normalized Points"
    ), row=1, col=2)

    bx_valid_base, by_valid_base, bz_valid_base = create_box_frame(
        [-pull_factor, -pull_factor, -pull_factor], 
        [pull_factor, pull_factor, pull_factor]
    )
    fig_compare.add_trace(go.Scatter3d(
        x=bx_valid_base, y=by_valid_base, z=bz_valid_base, mode='lines',
        line=dict(color='blue', width=1), name="Base Valid Region"
    ), row=1, col=2)

    fig_compare.add_trace(go.Scatter3d(
        x=bx_valid, y=by_valid, z=bz_valid, mode='lines',
        line=dict(color='green', width=1), name="Target [-1, 1] Region"
    ), row=1, col=2)

    fig_compare.update_layout(
        title=f"Normalization Comparison: Scene {scene} - {query_name}",
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data'),
        height=700,
        width=1200,
        showlegend=True,
        template="plotly_dark" 
    )

    output_html_compare = Path(f"normalization_comparison_{scene}.html")
    fig_compare.write_html(str(output_html_compare))
    logger.info(f"Comparison HTML successfully saved to {output_html_compare}")


if __name__ == "__main__":
    main()