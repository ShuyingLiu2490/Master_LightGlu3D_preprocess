import pycolmap
import numpy as np
from hloc.utils import viz_3d
from pathlib import Path
from utils.utils import qvec2rotmat
import plotly.io as pio
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

pio.renderers.default = "vscode"

def visualize_sfm_3d(sfm_dir: Path, scene: str, html_dir: Path, save_html: bool = True):
    '''
    Visualizes the 3D reconstruction of a given scene using pycolmap and hloc's viz_3d.
    Exports the reconstruction to a PLY file if it doesn't already exist.
    Saves an HTML file for 3D visualization if save_ply is True.
    Args:
        sfm_dir (Path): Directory containing the SfM reconstruction files.
        scene (str): Name of the scene to visualize.
        html_dir (Path): Directory to save the HTML visualization file.
        save_html (bool): Whether to save the HTML visualization.
    '''
    if not sfm_dir.exists():
        print(f"Error: Directory {str(sfm_dir)} does not exist!")
    else:
        reconstruction = pycolmap.Reconstruction(sfm_dir)
        ply_path = sfm_dir / f"reconstruction_{scene}.ply"

        # if not ply_path.exists():
        print(f"Exporting PLY model for scene {scene}...")
        reconstruction.export_PLY(str(ply_path))
        print(f"PLY model saved to {ply_path}.")
        # else:
        #     print(f"PLY model for scene {scene} already exists.")

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(
        fig, reconstruction, color='rgba(255,0,0,0.5)', name=f"triangulation of scene {scene}"
        )
    # fig.show()

    if save_html:
        html_path = html_dir / f"viz_{scene}.html"
        fig.write_html(str(html_path))
        print(f"HTML for 3D visualization saved to {html_path}.")

    return fig, reconstruction

def visualize_single_covisibility(
        sfm_dir: Path, 
        scene: str, 
        covisibility_results: dict, 
        query_cameras: dict,
        query_name: str, 
        html_dir: Path, 
        save_html: bool = True
        ):
    
    # Visualize the full SfM model first
    fig, reconstruction = visualize_sfm_3d(sfm_dir, scene, html_dir, save_html=False)

    if query_name not in covisibility_results:
        raise ValueError(f"{query_name} not found in covisibility_results")
    data = covisibility_results[query_name]

    # Color visiable 3D points as red
    xs, ys, zs = [], [], []

    for pid in data['unique_points']:
        pid = int(pid)
        if pid in reconstruction.points3D:
            xyz = reconstruction.points3D[pid].xyz
            xs.append(xyz[0])
            ys.append(xyz[1])
            zs.append(xyz[2])
        else:
            print(f"Warning: Point3D ID {pid} not found in reconstruction points3D.")

    fig.add_scatter3d(
        x=xs, y=ys, z=zs, mode='markers', marker=dict(size=3, color='green'), name="Covisible Points"
        )
    
    # Color reference images as blue
    cam_x, cam_y, cam_z = [], [], []

    for img_id in data['unique_images']:
        img_id = int(img_id)
        if img_id in reconstruction.images:
            image = reconstruction.images[img_id]
            cam_center = np.array(image.projection_center())

            cam_x.append(cam_center[0])
            cam_y.append(cam_center[1])
            cam_z.append(cam_center[2])
        else:
            print(f"Warning: Image ID {img_id} not found in reconstruction images.")

    fig.add_scatter3d(
        x=cam_x, y=cam_y, z=cam_z, mode='markers', marker=dict(size=6, color='blue'), name="Covisible Cameras"
        )
    
    # Color query image as green if query_cameras provided
    if query_cameras:
        if query_name not in query_cameras:
            raise ValueError(f"{query_name} not found in query_cameras")
        else:
            query_info = query_cameras[query_name]
            R = qvec2rotmat(query_info["qvec"])
            cam_center = -R.T @ query_info["tvec"]
            
            fig.add_scatter3d(
                x=[cam_center[0]],
                y=[cam_center[1]],
                z=[cam_center[2]],
                mode='markers',
                marker=dict(size=12, color='green'),
                name="Query Camera"
            )

    if save_html:
        html_path = html_dir / f"viz_{scene}_{query_name}.html"
        fig.write_html(str(html_path))
        print(f"Saved to {html_path}")
    else:
        fig.show()
    
    return fig

def visualize_2d3d_matches(
        gt_data, query_name, query_image_path, query_cameras=None, html_dir=None, plane_distance=0.5, save_html=False
        ):
    """
    gt_data:
        keypoints0: (N,2)
        keypoints1: (M,3)
        matches0: (N,)
        matches1: (M,)
    """
    
    keypoints2D = gt_data["keypoints0"]
    points3D = gt_data["keypoints1"]
    matches0 = gt_data["matches0"]
    
    valid_idx = np.where(matches0 != -1)[0]
    matched_3d_idx = matches0[valid_idx]

    if len(valid_idx) == 0:
        print(f"Warning: No valid matches found for {query_name}. Skipping visualization.")
        return None

    matched_2d = keypoints2D[valid_idx]
    matched_3d = points3D[matched_3d_idx]
    
    if query_cameras is not None:
        cam_info = query_cameras
        R = qvec2rotmat(cam_info["qvec"])
        t = cam_info["tvec"]

        cam_center = (-R.T @ t).reshape(3)
        cam_z = R.T[:, 2]
        cam_x = R.T[:, 0]
        cam_y = R.T[:, 1]
    else:
        cam_center = np.zeros(3)
        cam_z = np.array([0, 0, 1])
        cam_x = np.array([1, 0, 0])
        cam_y = np.array([0, 1, 0])

    img = Image.open(query_image_path)
    width, height = img.size
    f = query_cameras.get("params", [width, width])[0]

    norm_x = (matched_2d[:, 0] / width - 0.5)
    norm_y = (matched_2d[:, 1] / height - 0.5)

    plane_points = []
    for nx, ny in zip(norm_x, norm_y):
        p = (
            cam_center
            + cam_z * plane_distance
            + cam_x * (nx * width / f * plane_distance)
            + cam_y * (ny * height / f * plane_distance)
        )
        plane_points.append(p.flatten())

    plane_points = np.stack(plane_points)
    
    fig = go.Figure()
    # all 3D points
    fig.add_trace(go.Scatter3d(
        x=points3D[:, 0],
        y=points3D[:, 1],
        z=points3D[:, 2],
        mode='markers',
        marker=dict(size=2, color='lightgray'),
        name="All 3D Points"
    ))
    # matched 3D points
    fig.add_trace(go.Scatter3d(
        x=matched_3d[:, 0],
        y=matched_3d[:, 1],
        z=matched_3d[:, 2],
        mode='markers',
        marker=dict(size=2, color='red'),
        name="Matched 3D Points"
    ))

    # 2D keypoints
    fig.add_trace(go.Scatter3d(
        x=plane_points[:, 0],
        y=plane_points[:, 1],
        z=plane_points[:, 2],
        mode='markers',
        marker=dict(size=2, color='green'),
        name="2D Keypoints"
    ))

    for p2d, p3d in zip(plane_points, matched_3d):
        fig.add_trace(go.Scatter3d(
            x=[p2d[0], p3d[0]],
            y=[p2d[1], p3d[1]],
            z=[p2d[2], p3d[2]],
            mode='lines',
            line=dict(color='yellow', width=1),
            showlegend=False,
            opacity=0.5
        ))

    fig.add_trace(go.Scatter3d(
        x=[cam_center[0]],
        y=[cam_center[1]],
        z=[cam_center[2]],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name="Camera Center"
    ))

    fig.update_layout(
        scene=dict(aspectmode='data'),
        title=f"2D-3D Matches: {query_name}"
    )

    if save_html:
        if html_dir is None:
            html_dir = Path(".")
        html_dir.mkdir(parents=True, exist_ok=True)

        html_path = html_dir / f"2d3d_{query_name}.html"
        fig.write_html(str(html_path))
        print(f"Saved to {html_path}")
    else:
        fig.show()

    return fig