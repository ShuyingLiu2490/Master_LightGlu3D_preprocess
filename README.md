```

# File structure

LightGlu3D/
├── split_query_ref
├──── extract_query_sets.py         # Split images into queries and references
├── triangulation
├──── run_sfm_cpu.sh                # Shell script to run sbatch CPU job
├──── run_sfm_gpu.sh                # Shell script to run sbatch GPU job
├──── triangulation_cpu_steps.py    # Triangulate and visualize SfM models (CPU)
├──── triangulation_gpu_steps.py    # Feature extraction & matching (GPU)
├──── view_3d_rotate.py
├──── visual_sfm_3d.py              # The visualization ply and html files
├── covisibility
├──── covisibility_search_pipe.py   # Filter block for query-relevant 3D points and references
├──── check_covisibility_thres.py
├──── covisibility_search_pipe_aachen.py # Change because of the different structure for Aachen dataset
├── feature
├──── precompute_features.py        # Cache the averaged descriptors for 3D points
├──── precompute_features_aachen.py # Change because of the different structure for Aachen dataset
├── ground_truth
├──── generate_gt_pairs.py          # Base function to generate Ground Truth pairs, further applied as dataloader in gluefactory
├──── generate_gt_pairs_by_scene.py # Faster
├──── generate_gt_pairs_soft.py     # Save the GT pairs with soft threshold
├──── generate_ref_gt_pairs_from_hloc_aachen.py # Generate reference GT pairs from HLOC results on Aachen dataset
├── baseline
├──── nn_baseline.py                # NN (Nearest Neighbour), import from gluefactory
├──── rr_baseline.py                # RR (Rotate + Remove one coordinator)
├──── pr_baseline.py                # PR (Projection to Reference pose) 
├──── pr_baseline_change.py         # PRC (PR change, use query intrinsics and reference extrinsics to do projection)
├──── rn_baseline.py                # RN (Rotate + Normalization)
├──── trained_matcher.py            # Run the trained Lightglu3d model with dynamtic strategy
├──── feature_3d_compute_old.py     # Old experiment on projecting 3D to an image, for baseline use
├──── image_retrieval.py            # Old experiment on projecting 3D to an image, for baseline use
├──── preprocessing.ipynb           # Old experiment on projecting 3D to an image, for baseline use
├── visualization
├──── rerun_tools.py                # Help rurun file
├──── rerun_johanna.py              # Help rurun file
├──── visualize_normalization.py    # Visualize the quantile normalization of Lightglu3d
├──── visualize_matches.py          # Visualize the predicted matches from baselines or trained model
├──── visualize_no_gt.py            # Visualize the matches with GT = 0, to check the false positives
├──── visualize_gt.py               # Visualize the GT matches with soft threshold, to check the effect of soft threshold
├──── visualize_2d_keypoint.py      # Visualize the 2D keypoints, to check the abnormal keypoints
├──── visualize_pose_estimation.py  # Visualize the pose estimation results of Megadeth
├──── visualize_no_correct_match.py # Visualize the query images with no correct match, to check the failure cases
├── evaluation
├──── cambridge_selected.txt        # The selected scenes for evaluation on Cambridge dataset
├──── pose_estimation.py            # Evaluate the pose estimation results on Megadepth
├──── pose_estimation_cambridge.py  # Evaluate the pose estimation results on Cambridge
├──── pose_estimation_aachen.py     # Evaluate the pose estimation results on Aachen
├──── hloc_aachen_pipeline.py       # Run the original HLOC pipeline on Aachen dataset to get the reference file
├──── inference.py                  # Run inference with soft GT on Megadepth validation and test scenes
├──── inference_aachen.py           # Run inference with HLOC reference GT on Aachen dataset
├──── inference_match_performance_statistics.py # Statistics on threshold for ground truth and matching performance
├──── inference_query_match_statistics.py # Statistics on matching performance for each query image
├──── inference_reproj_statistics.py # Statistics on inlier reprojection error for each query image
├──── sigma_distribution.py         # Functions on the average sigma distribution
├──── inference_match_statistics.py # Statistics on query images with different number of matches and correct matches with dynamic strategy
├──utils
├──── utils.py
├── jupyter_pipeline
├──── 2d_3d_matching_test.ipynb
├──── run_2d3d_matching_visual.ipynb
├──── run_sfm_visualization.ipynb   # Notebook for SfM visualization (pre/post search)

---

# Merge with Jia in Github

# Berzelius cluster login
password: lsy20020409
code: (from NSC Berzelius:x_lishu)

cd matching/colla_preprocess/3d-2d-Matching-
cd matching/colla_gluefactory/glue-factory-2d3d-match

---

# Git operations for preprocess
# git init
# git remote add origin https://github.com/Jia0603/3d-2d-Matching-.git
# git fetch origin
# git checkout -b lsy-merged origin/lsy-merged
git checkout lsy-merged
git config --global user.name "Shuying Liu"
git config --global user.email "liushuying.blaise.2490@gmail.com"
git add .
git commit -m ""
git push origin lsy-merged --force

# Git operations for training
# git init
# git clone https://github.com/Jia0603/glue-factory-2d3d-match.git
git pull
git config --global user.name "Shuying Liu"
git config --global user.email "liushuying.blaise.2490@gmail.com"
git add .
git commit -m ""
git push


---

# Get GPU
interactive --gpus=1 -t 4:00:00
scontrol show job 16514559
# Get CPU (for triangulation)
interactive -p berzelius-cpu -t 4:00:00 --mem 128
# Check the project info
projinfo
# Check my use
squeue -u x_lishu
# cancel resource
scancel 15915922
# Check disk storage
nscquota

---

# Environment (install process in lsy-old)
mamba activate matchenv

---

# Aachen dataset image
unzip /proj/vlarsson/datasets/aachen_v1.1/images/database_and_query_images.zip -d /proj/vlarsson/outputs_aachen/aachen_images_unzip

# Dataset
/proj/vlarsson/datasets/aachen_v1.1
# Images
/proj/vlarsson/outputs_aachen/aachen_images_unzip
# Output
/proj/vlarsson/outputs_aachen

# Aachen dataset v1.1
unzip /proj/vlarsson/datasets/aachen_v1.1/aachen_v1_1.zip -d /proj/vlarsson/outputs_aachen/aachen_images_unzip_v1_1
# Additional images

rm -rf /proj/vlarsson/outputs_aachen/covisibility

---

# Split query images (split_query_ref)
# Change the way to get scene, from scene list or single scene for test
# Change to accpet arguments for ratios
# Change logger and tqdm

# Single scene
# Valid
python -m split_query_ref.extract_query_sets_re \
 --outputs  /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene 0022

# Test
python -m split_query_ref.extract_query_sets_re \
 --outputs  /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene 0025

# Train scene list
python -m split_query_ref.extract_query_sets_re \
 --outputs  /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt

# Output
ls /proj/vlarsson/users/x_lishu/colla_matching/outputs/query

---

# Triangulation

# GPU steps (feature extraction and matching)
python -m triangulation.triangulation_gpu_steps_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --query_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene 0022
python -m triangulation.triangulation_gpu_steps_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --query_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt

# CPU steps (triangulation)
python -m triangulation.triangulation_cpu_steps_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --html_save_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/saved_html_visual \
 --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt

python -m triangulation.triangulation_cpu_steps_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs  /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --html_save_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/saved_html_visual \
 --scene 0022

# Output
ls /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation

# Special triangulation for Aachen
python -m triangulation.triangulation_gpu_steps_aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --outputs /proj/vlarsson/outputs_aachen/sfm
python -m triangulation.triangulation_cpu_steps_aachen \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --outputs /proj/vlarsson/outputs_aachen/sfm

---

# Covisibility

covisibility_search_pipe_re.py

python -m covisibility.covisibility_search_pipe_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs /proj/vlarsson/users/x_lishu/colla_matching/outputs/covisibility \
 --sfm_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --query_list  /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt

python -m covisibility.covisibility_search_pipe_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs /proj/vlarsson/users/x_lishu/colla_matching/outputs/covisibility \
 --sfm_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --query_list  /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --scene 0022

# Covisibility process for Aachen dataset
python -m covisibility.covisibility_search_pipe_aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --outputs /proj/vlarsson/outputs_aachen/covisibility \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --v1_1_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip_v1_1 \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
 --query_dir /proj/vlarsson/datasets/aachen_v1.1/queries

---

# Feature computation

python -m feature.precompute_features_re \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --outputs /proj/vlarsson/users/x_lishu/colla_matching/outputs/covisibility \
 --sfm_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt

python -m feature.precompute_features_aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --outputs /proj/vlarsson/outputs_aachen/covisibility \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm

---

# Ground Truth

python -m ground_truth.generate_gt_pairs_re \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --query_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --sfm_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --feature_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/covisibility \
 --scene_list /home/x_lishu/matching/glue-factory/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean_try.txt

python -m ground_truth.generate_gt_pairs_re \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --query_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
 --sfm_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
 --feature_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/covisibility \
 --scene 0022

# Soft threshold
# Defult 
python -m ground_truth.generate_gt_pairs_soft \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --query_dir /proj/vlarsson/outputs/query_sets \
 --sfm_dir /proj/vlarsson/outputs/sfm \
 --feature_dir /proj/vlarsson/outputs/midterm_results \
 --scene 0036

# output
ls /proj/vlarsson/users/x_lishu/colla_matching/outputs/feature

---

# Train

# Train with a adaptor MLP
# Only train on adaptor MLP, position, Last two layers of Lightglue
# v1: only adaptor and position
# v2: adaptor, position and two last layers, lr = 1e-4
python -m gluefactory.train lightglue_adapt_v2     --conf gluefactory/configs/2d_3d_lightglue_SP_finetune.yaml
# v3: adaptor, position and two last layers, soft threshould on gt, lr = 1e-5
python -m gluefactory.train lightglue_adapt_v3     --conf gluefactory/configs/2d_3d_lightglue_adapt_SP_finetune.yaml

# Tensorboard
tensorboard --logdir ~/matching/colla_gluefactory/glue-factory-2d3d-match/outputs/training/lightglue_adapt_v3 --port 6008

---

# NN baseline grid search
python -m gluefactory.run_nn_baseline_grid

# Change the path inside
python -m gluefactory.run_nn_baseline

# RR baseline
python -m baseline.rr_baseline \
 --dataset  /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir  /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene_list /proj/vlarsson/outputs/splits/test.txt

# PR baseline
python -m baseline.pr_baseline \
 --dataset  /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir /proj/vlarsson/outputs/query_sets \
 --sfm_dir /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene_list /proj/vlarsson/outputs/splits/test.txt

# Changed PR baseline
# Use query intrinsics
python -m baseline.pr_baseline_change \
 --dataset  /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir /proj/vlarsson/outputs/query_sets \
 --sfm_dir /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene_list /proj/vlarsson/outputs/splits/test.txt

# RN baselin
python -m baseline.rn_baseline \
 --dataset  /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir /proj/vlarsson/outputs/query_sets \
 --sfm_dir /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene_list /proj/vlarsson/outputs/splits/test.txt

# Change the ground truth threshold, check the matching ability for baselines
# Since before don't write argument for soft threshold, just change the parameters in ground truth soft function directly
# Try threshold combination (5.0 / 8.0), (3.0 / 8.0), (6.0 / 12.0) in megadepth validation scenes
# Change it back to (3.0 / 5.0) after
python -m baseline.rr_baseline \
 --dataset  /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir  /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene_list /proj/vlarsson/outputs/splits/val.txt

python -m baseline.pr_baseline \
 --dataset  /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir /proj/vlarsson/outputs/query_sets \
 --sfm_dir /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene_list /proj/vlarsson/outputs/splits/val.txt


---

# Match metrics for inference
export PYTHONPATH="/home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match:$PYTHONPATH"
python -m evaluation.inference \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir /proj/vlarsson/outputs/query_sets \
 --sfm_dir /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene_list /proj/vlarsson/outputs/splits/val.txt \
 --method TRAIN \
 --checkpoint /home/x_lishu/matching/network_weights/nll.5_30ep_checkpoint.tar

python -m evaluation.inference \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir /proj/vlarsson/outputs/query_sets \
 --sfm_dir /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene_list /proj/vlarsson/outputs/splits/test.txt \
 --method TRAIN \
 --checkpoint /home/x_lishu/matching/network_weights/nll.5_30ep_checkpoint.tar



---

# Visualization

# Visulaization of normalization
python -m visualization.visualize_normalization \
  --covisibility_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/covisibility \
  --query_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/query \
  --sfm_dir /proj/vlarsson/users/x_lishu/colla_matching/outputs/triangulation \
  --scene 0022

# Visulization of matches
python -m visualization.visualize_matches \
 --dataset_type aachen \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --scene night \
 --method TRAIN \
 --checkpoint /home/x_lishu/matching/network_weights/checkpoint_best_gt_38_gau_0.005.tar \
 --query_name 2835868540_572241d9f7_o.jpg


python -m visualization.visualize_matches \
 --dataset_type aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --covisibility_dir /proj/vlarsson/outputs_aachen/covisibility \
 --query_dir /proj/vlarsson/datasets/aachen_v1.1/queries \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
  --hloc_reference /proj/vlarsson/outputs_aachen/hloc_outputs/Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt \
 --scene night \
 --method TRAIN \
 --checkpoint /home/x_lishu/matching/network_weights/checkpoint_best_debug.tar \
 --query_name query/night/nexus5x_additional_night/IMG_20170702_003514.jpg

python -m visualization.visualize_matches \
 --dataset_type aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --covisibility_dir /proj/vlarsson/outputs_aachen/covisibility \
 --query_dir /proj/vlarsson/datasets/aachen_v1.1/queries \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
  --hloc_reference /proj/vlarsson/outputs_aachen/hloc_outputs/Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt \
 --scene night \
 --method TRAIN \
 --checkpoint /home/x_lishu/matching/network_weights/checkpoint_best_debug.tar \
 --query_name query/night/nexus5x_additional_night/IMG_20170702_003514.jpg

python -m visualization.visualize_matches \
 --dataset_type aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --covisibility_dir /proj/vlarsson/outputs_aachen/covisibility \
 --query_dir /proj/vlarsson/datasets/aachen_v1.1/queries \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
  --hloc_reference /proj/vlarsson/outputs_aachen/hloc_outputs/Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt \
 --scene night \
 --method PR \
 --query_name query/night/nexus5x_additional_night/IMG_20170702_003150.jpg

query/night/nexus5x_additional_night/IMG_20170702_003150.jpg

python -m visualization.visualize_no_correct_match \
 --dataset_type aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --covisibility_dir /proj/vlarsson/outputs_aachen/covisibility \
 --query_dir /proj/vlarsson/datasets/aachen_v1.1/queries \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
  --hloc_reference /proj/vlarsson/outputs_aachen/hloc_outputs/Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt \
 --scene night \
 --method TRAIN \
 --checkpoint /home/x_lishu/matching/network_weights/checkpoint_best_debug.tar

query/night/nexus5x_additional_night/IMG_20170702_003514.jpg
query/night/nexus5x_additional_night/IMG_20170702_003150.jpg

# Visualize training
# Chnage this to add the custom path before visualizing TRAIN
export PYTHONPATH="/home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match:$PYTHONPATH"
# Put the .tar file into visulization.network_weights folder
python -m visualization.visualize_matches \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
 --scene 0022 \
 --method TRAIN \
 --checkpoint /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/visualization/network_weights/checkpoint_best_clean.tar

# Visualize ground truth = 0
# Before becasue added the extra line so it will collapse
python -m visualization.visualize_no_gt \
  --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
  --covisibility_dir /proj/vlarsson/outputs/midterm_results \
  --query_dir  /proj/vlarsson/outputs/query_sets \
  --sfm_dir  /proj/vlarsson/outputs/sfm \
  --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
  --scene 0022

# Visualize abnormal superpoint keypoints
python -m visualization.visualize_2d_keypoint \
  --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
  --query_dir /proj/vlarsson/outputs/query_sets \
  --scene 0022

# Visualize ground truth with soft threshold
python -m visualization.visualize_gt \
  --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
  --covisibility_dir /proj/vlarsson/outputs/midterm_results \
  --query_dir  /proj/vlarsson/outputs/query_sets \
  --sfm_dir  /proj/vlarsson/outputs/sfm \
  --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
  --scene 0022

# Visualization of pose estimation
python -m visualization.visualize_pose_estimation \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --scene 0022 \
 --method NN \
 --max_error 12

---

# Estimation
# Change this to add the custom path before visualizing TRAIN
export PYTHONPATH="/home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match:$PYTHONPATH"

# Get some scalers for val scenes. Apply them to get correct pose estiamtion on megadpeth val
python -m evaluation.pose_estimation \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --query_dir  /proj/vlarsson/outputs/query_sets \
 --sfm_dir  /proj/vlarsson/outputs/sfm \
 --scene_list /proj/vlarsson/outputs/splits/val.txt \
 --method TRAIN \
 --checkpoint /home/x_lishu/matching/network_weights/checkpoint_best_gt_38_gau_0.002.tar \
 --max_error 12


# Cambridge
python -m evaluation.pose_estimation_cambridge \
 --dataset /proj/vlarsson/datasets/cambridge \
 --covisibility_dir /proj/vlarsson/outputs_cambridge/midterm_results \
 --query_dir /proj/vlarsson/datasets/cambridge/CambridgeLandmarks_Colmap_Retriangulated_1024px \
 --sfm_dir /proj/vlarsson/outputs_cambridge/sfm \
 --scene_list /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/cambridge_selected.txt \
 --method PR \
 --max_error 12

python -m evaluation.pose_estimation_cambridge \
 --dataset /proj/vlarsson/datasets/cambridge \
 --covisibility_dir /proj/vlarsson/outputs_cambridge/midterm_results \
 --query_dir /proj/vlarsson/datasets/cambridge/CambridgeLandmarks_Colmap_Retriangulated_1024px \
 --sfm_dir /proj/vlarsson/outputs_cambridge/sfm \
 --scene_list /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/cambridge_selected.txt \
 --method TRAIN \
 --checkpoint /home/x_lishu/matching/network_weights/checkpoint_best_debug.tar \
 --max_error 12

python -m evaluation.pose_estimation_cambridge \
 --dataset /proj/vlarsson/datasets/cambridge \
 --covisibility_dir /proj/vlarsson/outputs_cambridge/midterm_results \
 --query_dir /proj/vlarsson/datasets/cambridge/CambridgeLandmarks_Colmap_Retriangulated_1024px \
 --sfm_dir /proj/vlarsson/outputs_cambridge/sfm \
 --scene_list /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/cambridge_selected.txt \
 --method HLOC \
 --max_error 12

python -m evaluation.pose_estimation_cambridge \
 --dataset /proj/vlarsson/datasets/cambridge \
 --covisibility_dir /proj/vlarsson/outputs_cambridge/midterm_results \
 --query_dir /proj/vlarsson/datasets/cambridge/CambridgeLandmarks_Colmap_Retriangulated_1024px \
 --sfm_dir /proj/vlarsson/outputs_cambridge/sfm \
 --scene_list /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/cambridge_selected.txt \
 --method ADAPT \
 --checkpoint /home/x_lishu/matching/network_weights/checkpoint_best_adapt.tar \
 --max_error 12


# Aachen
# Fristly, run hloc original pipeline and get the reference file. 
python -m evaluation.hloc_aachen_pipeline \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --outputs /proj/vlarsson/outputs_aachen/hloc_outputs

# Build SfM
python -m triangulation.triangulation_aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --outputs /proj/vlarsson/outputs_aachen/sfm

# Extract query feature

# Covisibility search on sfm
python -m covisibility.covisibility_search_pipe_aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --outputs /proj/vlarsson/outputs_aachen/covisibility \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
 --query_dir /proj/vlarsson/datasets/aachen_v1.1/queries

# Compute SfM features
python -m feature.precompute_features_aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --outputs /proj/vlarsson/outputs_aachen/covisibility \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm

# Absolute pose estimation
export PYTHONPATH="/home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match:$PYTHONPATH"
python -m evaluation.pose_estimation_aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --covisibility_dir /proj/vlarsson/outputs_aachen/covisibility \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
 --query_dir /proj/vlarsson/datasets/aachen_v1.1/queries \
 --outputs /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/outputs \
 --method TRAIN \
 --checkpoint /home/x_lishu/matching/network_weights/nll.5_30ep_checkpoint.tar \
 --max_error 12 \
 --hloc_reference /proj/vlarsson/outputs_aachen/hloc_outputs/Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt

python -m evaluation.pose_estimation_aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --covisibility_dir /proj/vlarsson/outputs_aachen/covisibility \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
 --query_dir /proj/vlarsson/datasets/aachen_v1.1/queries \
 --outputs /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/outputs \
 --method PRC \
 --max_error 12 \
 --hloc_reference /proj/vlarsson/outputs_aachen/hloc_outputs/Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt



# Add statictics in covisibility
python -m covisibility.covisibility_search_pipe_aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --image_dir /proj/vlarsson/outputs_aachen/aachen_images_unzip/images_upright \
 --outputs /proj/vlarsson/outputs_aachen/covisibility \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
 --query_dir /proj/vlarsson/datasets/aachen_v1.1/queries

# Use hloc to get ref gt
python -m ground_truth.generate_ref_gt_pairs_from_hloc_aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --query_dir /proj/vlarsson/datasets/aachen_v1.1/queries \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
 --covisibility_dir /proj/vlarsson/outputs_aachen/covisibility \
 --outputs /proj/vlarsson/outputs_aachen/gt \
 --hloc_reference /proj/vlarsson/outputs_aachen/hloc_outputs/Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt \
 --pos_reproj_thresh 3.0 \
 --neg_reproj_thresh 5.0 

# Inference to get match percision and recall by comparing with ref gt
python -m evaluation.inference_aachen \
  --dataset /proj/vlarsson/datasets/aachen_v1.1 \
  --covisibility_dir /proj/vlarsson/outputs_aachen/covisibility \
  --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
  --gt_path /proj/vlarsson/outputs_aachen/gt/aachen_ref_ground_truth.pkl \
  --method TRAIN \
  --checkpoint /home/x_lishu/matching/network_weights/nll.5_30ep_checkpoint.tar

python -m evaluation.inference_aachen \
  --dataset /proj/vlarsson/datasets/aachen_v1.1 \
  --covisibility_dir /proj/vlarsson/outputs_aachen/covisibility \
  --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
  --gt_path /proj/vlarsson/outputs_aachen/gt/aachen_ref_ground_truth.pkl \
  --method PRC

---

# Statistics 
python -m evaluation.inference_match_performance_statistics \
  --dataset_type aachen \
  --dataset /proj/vlarsson/datasets/aachen_v1.1 \
  --covisibility_dir /proj/vlarsson/outputs_aachen/covisibility \
  --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
  --outputs /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/statistics \
  --hloc_reference /proj/vlarsson/outputs_aachen/hloc_outputs/Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt \
  --checkpoint_train /home/x_lishu/matching/network_weights/checkpoint_best_debug.tar \
  --checkpoint_adapt /home/x_lishu/matching/network_weights/checkpoint_best_adapt.tar

 python -m evaluation.inference_match_performance_statistics \
   --dataset_type megadepth \
   --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
   --covisibility_dir /proj/vlarsson/outputs/midterm_results \
   --sfm_dir /proj/vlarsson/outputs/sfm \
   --query_dir /proj/vlarsson/outputs/query_sets \
   --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
   --scene_list /proj/vlarsson/outputs/splits/val.txt \
   --outputs /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation \
   --checkpoint_train /home/x_lishu/matching/network_weights/checkpoint_best_debug.tar \
   --checkpoint_adapt /home/x_lishu/matching/network_weights/checkpoint_best_adapt.tar

python -m evaluation.inference_query_match_statistics \
 --dataset_type aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
  --covisibility_dir /proj/vlarsson/outputs_aachen/covisibility \
  --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
  --outputs /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/statistics \
  --checkpoint_train /home/x_lishu/matching/network_weights/nll.5_30ep_checkpoint.tar \
  --checkpoint_adapt /home/x_lishu/matching/network_weights/checkpoint_best_adapt.tar \
  --hloc_reference /proj/vlarsson/outputs_aachen/hloc_outputs/Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt

python -m evaluation.inference_query_match_statistics \
   --dataset_type megadepth \
   --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
   --covisibility_dir /proj/vlarsson/outputs/midterm_results \
   --sfm_dir /proj/vlarsson/outputs/sfm \
  --outputs /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/statistics \
  --checkpoint_train /home/x_lishu/matching/network_weights/checkpoint_best_debug.tar \
  --query_dir /proj/vlarsson/outputs/query_sets \
  --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted \
  --scene_list /proj/vlarsson/outputs/splits/val.txt

python -m evaluation.inference_reproj_statistics \
   --dataset_type megadepth \
   --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
   --covisibility_dir /proj/vlarsson/outputs/midterm_results \
   --sfm_dir /proj/vlarsson/outputs/sfm \
  --outputs /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/statistics \
  --checkpoint_train /home/x_lishu/matching/network_weights/checkpoint_best_debug.tar \
  --checkpoint_adapt /home/x_lishu/matching/network_weights/checkpoint_best_adapt.tar \
  --query_dir /proj/vlarsson/outputs/query_sets \
  --scene_list /proj/vlarsson/outputs/splits/val.txt

python -m evaluation.inference_reproj_statistics \
   --dataset_type aachen \
   --dataset /proj/vlarsson/datasets/aachen_v1.1 \
   --covisibility_dir /proj/vlarsson/outputs_aachen/covisibility \
   --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
  --outputs /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/statistics \
  --checkpoint_train /home/x_lishu/matching/network_weights/checkpoint_best_gt_38_gau_0.005.tar \
  --checkpoint_adapt /home/x_lishu/matching/network_weights/checkpoint_best_adapt.tar \
  --hloc_reference /proj/vlarsson/outputs_aachen/hloc_outputs/Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt

# absolute matches
python -m evaluation.inference_match_statistics \
 --dataset_type megadepth \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --sfm_dir /proj/vlarsson/outputs/sfm \
 --outputs /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/statistics \
 --checkpoint_train /home/x_lishu/matching/network_weights/nll.5_30ep_checkpoint.tar \
 --query_dir /proj/vlarsson/outputs/query_sets \
 --scene_list /proj/vlarsson/outputs/splits/val.txt \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted

python -m evaluation.inference_match_statistics \
 --dataset_type megadepth \
 --dataset /proj/vlarsson/datasets/megadepth/Undistorted_SfM \
 --covisibility_dir /proj/vlarsson/outputs/midterm_results \
 --sfm_dir /proj/vlarsson/outputs/sfm \
 --outputs /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/statistics \
 --checkpoint_train /home/x_lishu/matching/network_weights/nll.5_30ep_checkpoint.tar \
 --query_dir /proj/vlarsson/outputs/query_sets \
 --scene_list /proj/vlarsson/outputs/splits/test.txt \
 --depth_dir /proj/vlarsson/datasets/megadepth/depth_undistorted

# Change filter threshold as 0.05, 0.025, 0.015
# Change to dynamic threshold
python -m evaluation.inference_match_statistics \
 --dataset_type aachen \
 --dataset /proj/vlarsson/datasets/aachen_v1.1 \
 --covisibility_dir /proj/vlarsson/outputs_aachen/covisibility \
 --sfm_dir /proj/vlarsson/outputs_aachen/sfm \
 --outputs /home/x_lishu/matching/colla_preprocess/3d-2d-Matching-/evaluation/statistics \
 --checkpoint_train /home/x_lishu/matching/network_weights/nll.5_30ep_checkpoint.tar \
 --hloc_reference /proj/vlarsson/outputs_aachen/hloc_outputs/Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt \
 --min_matches 400

'''