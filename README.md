# Final Preprocess and Evaluation

## Introduction

Here you can find the preprocess part of LightGlu3D, including split the query and reference, SfM for the scenes, covisibility search, 3D feature calculation. 

The three baselines and the loading of the trained model. Two baselines for matching, MNN and PR. One baselines for localization, HLOC.

The evaluation on matching performance and absolute pose estimation. 

And some visualization in rerun.



You could find the detailed in thesis here: 

LightGlu3D training repo: 



## Install the environment

Create a mamba env for this project:

```
mamba create --name matchenv python=3.10
mamba activate matchenv
```

Install the following repos:

1. LightGlue

   ```
   git clone https://github.com/cvg/LightGlue.git
   cd LightGlue
   python -m pip install -e .
   ```

2. GlueFactory

   ```
   git clone https://github.com/cvg/glue-factory.git
   cd glue-factory
   python3 -m pip install -e .
   ```

3. HLOC

   ```
   git clone https://github.com/cvg/Hierarchical-Localization.git
   cd Hierarchical-Localization
   python -m pip install -e .
   git submodule update --init --recursive
   ```

   

## Install the datasets

1. MegaDepth
2. Cambridge Landmarks
3. Aachen Day&Night v1.1



## File Structure

```
├──preprocess_Megadepth
│   ├──extract_query_sets.py 			# Split dataset into query and reference
│   ├──triangulation_gpu_steps.py 		# Feature extraction and matching preparson for triangulation
│   ├──triangulation_cpu_steps.py 		# Triangulation
│   ├──covisibility_search_pipe.py 		# Find visible 3D points for each query based on covisibility 		
│										# expansion
│   ├──precompute_features.py 			# Calculate averaged 3D features
│   └──generate_gt_pairs_by_scene.py 	# GT calculation functions
│ 
├──preprocess_extra 					# extra preprocess code for Cambridge and Aachen
│   ├──triangulation_aachen.py 			# Triangulation Aachen in original SIFT coordinate
│   ├──covisibility_search_pipe_aachen.py # Covisibility search on Aachen
│   └──precompute_features_aachen.py 	# Calculated averaged 3D features on Aachen
│ 
├──baselines_and_trained_matcher
│   ├──mnn_baseline.py 					# Mutual Nearest Neighbour (MNN) matching baseline
│   ├──pr_lg_baseline.py 				# Projection Reference LightGlue (PR) matching baseline
│   ├──lightglu3d_bicross.py 			# Final LightGlu3D matcher benchmark
│   ├──trained_matcher.py 				# Use the trained matcher and dynamic strategy
│   └──network_weights 					# Put the trained weights here
│
├──evaluation
│   ├──cambridge_selected.txt 			# Scene list for Cambridge Landmarks
│   ├──inference.py 					# Match performance for all matchers on MegaDepth
│   └──pose_estimation.py 				# Absolute pose estimation for all matchers on all datasets
│
├──visualization
│   ├──visualize_gt.py 					# Visualization of the soft threshold effect
│   ├──visualize_normalization.py 		# Visualization of 3D normalization
│   ├──visualize_matches.py 			# Visualziation of the matching
│   └──visualize_no_match.py 			# Visualization the no matching case
```



## Preprocess

First is to generate the trained data from MegaDepth.

### Split uery and reference

In MegaDepth, we have train.txt, valid.txt, and test.txt. Also, can set small scenes try.

Some arguments could try: sample_ratio: how many 3D points are considered from the original SIFT 3D model, query_ratio: how many images could be query.

```
python -m preprocess_Megadepth.extract_query_sets \
 --outputs [YOUR_MEGADEPTH_OUTPUT_FOLDER]/query \
 --scene_list [SCENES_TXT_FILE_PATH]
```



### Triangulation

Triangulation based on SuperPoint feature and LightGlue matching.  This part the code is separated in GPU step and CPU step.

GPU part is the feature matching. CPU part is the triangulation.

Some arguments could try in GPU part: min_overlap and max_overlap: the score threshold from the overlap matrix in scene_info.

```
python -m preprocess_Megadepth.triangulation_gpu_steps \
 --dataset [YOUR_MEGADPETH_FOLDER]/Undistorted_SfM \
 --outputs [YOUR_MEGADEPTH_OUTPUT_FOLDER]/triangulation \
 --query_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/query \
 --scene_list [SCENES_TXT_FILE_PATH]
 
python -m preprocess_Megadepth.triangulation_cpu_steps \
 --dataset [YOUR_MEGADPETH_FOLDER]/Undistorted_SfM \
 --outputs [YOUR_MEGADEPTH_OUTPUT_FOLDER]/triangulation \
 --html_save_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/saved_html_visual \
 --scene_list [SCENES_TXT_FILE_PATH]
```



### Covisibility search

Find the visible 3D points for each query.

Some arguments could try: pruning: pruning factor for covisible overlap.

```
python -m preprocess_Megadepth.covisibility_search_pipe \
 --dataset [YOUR_MEGADPETH_FOLDER]/Undistorted_SfM \
 --outputs [YOUR_MEGADEPTH_OUTPUT_FOLDER]/covisibility \
 --sfm_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/triangulation \
 --query_list [YOUR_MEGADEPTH_OUTPUT_FOLDER]/query \
 --scene_list [SCENES_TXT_FILE_PATH]
```



### Calculate 3D features

Average the 2D features from track of each 3D points to get the 3D features.

```
python -m preprocess_Megadepth.precompute_features \
 --dataset [YOUR_MEGADPETH_FOLDER]/Undistorted_SfM \
 --outputs [YOUR_MEGADEPTH_OUTPUT_FOLDER]/covisibility \
 --sfm_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/triangulation \
 --scene_list [SCENES_TXT_FILE_PATH]
```



### Ground Truth Calculation

The ground truth calculation functions could be found in `preprocess_Megadepth.generate_gt_pairs_by_scene`, including the calculation with and without the ignore label. Both consider reprojection error and depth error.



Similarly, we preprocess for Cambridge and Aachen.



### Preprocess for Aachen

In Aachen, it has already separated the query and reference. So skip that step.

Start from the triangulation. (The dataset we put is not in a good structure, so we add `image_dir` here.)

```
python -m preprocess_extra.triangulation_aachen \
 --dataset [YOUR_AACHEN_FOLDER](aachen_v1.1) \
 --image_dir [YOUR_AACHEN_FOLDER]/aachen_images_unzip/images_upright \
 --outputs [YOUR_AACHEN_OUTPUT_FOLDER]/triangulation
```



Then is the covisibility search.

```
python -m preprocess_extra.covisibility_search_pipe_aachen \
 --dataset [YOUR_AACHEN_FOLDER](aachen_v1.1) \
 --outputs [YOUR_AACHEN_OUTPUT_FOLDER]/covisibility \
 --image_dir [YOUR_AACHEN_FOLDER]/aachen_images_unzip/images_upright \
 --sfm_dir [YOUR_AACHEN_OUTPUT_FOLDER]/triangulation\
 --query_dir [YOUR_AACHEN_FOLDER]/queries
```



Calculate the 3D features

```
python -m preprocess_extra.precompute_features_aachen \
 --dataset [YOUR_AACHEN_FOLDER](aachen_v1.1) \
 --outputs [YOUR_AACHEN_OUTPUT_FOLDER]/covisibility \
 --sfm_dir [YOUR_AACHEN_OUTPUT_FOLDER]/triangulation
```



## Matchers

### Baselines

1. Mutual Nearest Neighbor (MNN): `compute_nn_baseline()` function in `mnn_baseline.py`.

2. Projection Reference LightGlue (PR): `compute_pr_baseline()` function in `pr_lg_baseline.py`.
3. Fair HLOC: `compute_hloc_baseline()` function in `hloc_fair_baseline.py`, only use for visual localization.



### Trained matcher

Inference LightGlu3D structure in `lightglu3d_bicross.py`.

Normally loading the trained matcher LightGlu3D: use `load_trained_lightglu3d()` and `compute_trained_lightglu3d()` to load the matcher, get the predicted matching and the score matrix.

With the dynamic greedy filter threshold strategy: use `filter_matches_greedy()` and `compute_trained_lightglu3d_greedy_dynamic()` to get more matching.



## Evaluation

### Matching performance

Run the MNN and PR baselines and trained LightGlu3D on MegaDepth validation and test scenes to get the match precision and recall.

1. MNN:

   ```
   python -m evaluation.inference_megadepth \
    --dataset [YOUR_MEGADPETH_FOLDER]/Undistorted_SfM \
    --covisibility_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/covisibility \
    --query_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/query \
    --sfm_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/triangulation \
    --depth_dir [YOUR_MEGADPETH_FOLDER]/depth_undistorted \
    --scene_list [SCENES_TXT_FILE_PATH]
    --method MNN
   ```

   

2. PR:

   ```
   python -m evaluation.inference_megadepth \
    --dataset [YOUR_MEGADPETH_FOLDER]/Undistorted_SfM \
    --covisibility_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/covisibility \
    --query_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/query \
    --sfm_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/triangulation \
    --depth_dir [YOUR_MEGADPETH_FOLDER]/depth_undistorted \
    --scene_list [SCENES_TXT_FILE_PATH]
    --method PR
   ```

   

3. LightGlu3D

   Need to have the trained weights .tar file. The `filter_threshold` argument could be changed.

   ```
   python -m evaluation.inference_megadepth \
    --dataset [YOUR_MEGADPETH_FOLDER]/Undistorted_SfM \
    --covisibility_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/covisibility \
    --query_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/query \
    --sfm_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/triangulation \
    --depth_dir [YOUR_MEGADPETH_FOLDER]/depth_undistorted \
    --scene_list [SCENES_TXT_FILE_PATH] \
    --method TRAIN \
    --checkpoint [YOUR_TRAINED_WEIGHTS](.tar) \
    --filter_threshold 0.05
   ```

   

Results:

In MegaDepth validation scenes:

| Method                          | Match Precision (%) | Match Recall (%) |
| ------------------------------- | ------------------- | ---------------- |
| MNN                             | 0.5574              | 0.4058           |
| PR                              | 0.7689              | 0.6765           |
| LightGlu3D (k=0.3, lambda=0.05) | 0.7883              | 0.8116           |
| LightGlu3D (k=0.3, lambda=0.1)  | 0.7955              | 0.8088           |
| LightGlu3D (k=0.5, lambda=0.05) | 0.7772              | 0.8136           |
| LightGlu3D (k=0.5, lambda=0.1)  | 0.7850              | 0.8117           |

In MegaDepth test scenes:

| Method                          | Match Precision (%) | Match Recall (%) |
| ------------------------------- | ------------------- | ---------------- |
| MNN                             | 0.6845              | 0.5434           |
| PR                              | 0.8592              | 0.8455           |
| LightGlu3D (k=0.3, lambda=0.05) | 0.8534              | 0.8852           |
| LightGlu3D (k=0.3, lambda=0.1)  | 0.8599              | 0.8833           |
| LightGlu3D (k=0.5, lambda=0.05) | 0.8422              | 0.8863           |
| LightGlu3D (k=0.5, lambda=0.1)  | 0.8496              | 0.8851           |



### Visual localization

Run the MNN, PR, HLOC baselines and trained LightGlu3D on Cambridge Landmarks and Aachen to get the absolute pose estimation.

#### Cambridge Landmarks

For Cambridge, could select the processed scene in `cambridge_selected.txt`.

1. MNN:

   ```
   python -m evaluation.pose_estimation \
    --dataset_type cambridge \
    --dataset [YOUR_CAMBRIDGE_FOLDER](cambridge) \
    --covisibility_dir [YOUR_CAMBRIDGE_OUTPUT_FOLDER]/covisibility \
    --query_dir [YOUR_CAMBRIDGE_FOLDER]/CambridgeLandmarks_Colmap_Retriangulated_1024px \
    --sfm_dir [YOUR_CAMBRIDGE_OUTPUT_FOLDER]/triangulation \
    --scene_list evaluation/cambridge_selected.txt \
    --method MNN
   ```

   

2. PR:

   ```
   python -m evaluation.pose_estimation \
    --dataset_type cambridge \
    --dataset [YOUR_CAMBRIDGE_FOLDER](cambridge) \
    --covisibility_dir [YOUR_CAMBRIDGE_OUTPUT_FOLDER]/covisibility \
    --query_dir [YOUR_CAMBRIDGE_FOLDER]/CambridgeLandmarks_Colmap_Retriangulated_1024px \
    --sfm_dir [YOUR_CAMBRIDGE_OUTPUT_FOLDER]/triangulation \
    --scene_list evaluation/cambridge_selected.txt \
    --method PR
   ```

   

3. HLOC:

   ```
   python -m evaluation.pose_estimation \
    --dataset_type cambridge \
    --dataset [YOUR_CAMBRIDGE_FOLDER](cambridge) \
    --covisibility_dir [YOUR_CAMBRIDGE_OUTPUT_FOLDER]/covisibility \
    --query_dir [YOUR_CAMBRIDGE_FOLDER]/CambridgeLandmarks_Colmap_Retriangulated_1024px \
    --sfm_dir [YOUR_CAMBRIDGE_OUTPUT_FOLDER]/triangulation \
    --scene_list evaluation/cambridge_selected.txt \
    --method HLOC
   ```

   

4. LightGlu3D:

   ```
   python -m evaluation.pose_estimation \
    --dataset_type cambridge \
    --dataset [YOUR_CAMBRIDGE_FOLDER](cambridge) \
    --covisibility_dir [YOUR_CAMBRIDGE_OUTPUT_FOLDER]/covisibility \
    --query_dir [YOUR_CAMBRIDGE_FOLDER]/CambridgeLandmarks_Colmap_Retriangulated_1024px \
    --sfm_dir [YOUR_CAMBRIDGE_OUTPUT_FOLDER]/triangulation \
    --scene_list evaluation/cambridge_selected.txt \
    --method TRAIN /
    --checkpoint [YOUR_TRAINED_WEIGHTS](.tar) \
    --filter_threshold 0.05
   ```

   

Results:

Median translation and rotation error:

Each block in this format: Median Trans Error / Median Rot Error

| Method                     | KingsCollege | ShopFacade | StMarysChurch | GreatCourt | OldHospital |
| -------------------------- | ------------ | ---------- | ------------- | ---------- | ----------- |
| MNN                        |              |            |               |            |             |
| PR                         |              |            |               |            |             |
| HLOC                       |              |            |               |            |             |
| LightGlu3D (k=0.3, λ=0.05) |              |            |               |            |             |
| LightGlu3D (k=0.3, λ=0.1)  |              |            |               |            |             |
| LightGlu3D (k=0.5, λ=0.05) |              |            |               |            |             |
| LightGlu3D (k=0.5, λ=0.1)  |              |            |               |            |             |



AUC of translation and rotation error:

Each block is in this format: (0.25m, 2◦) / (0.05m, 5◦) / (5.00m, 10◦).

| Method                     | KingsCollege | ShopFacade | StMarysChurch | GreatCourt | OldHospital |
| -------------------------- | ------------ | ---------- | ------------- | ---------- | ----------- |
| MNN                        |              |            |               |            |             |
| PR                         |              |            |               |            |             |
| HLOC                       |              |            |               |            |             |
| LightGlu3D (k=0.3, λ=0.05) |              |            |               |            |             |
| LightGlu3D (k=0.3, λ=0.1)  |              |            |               |            |             |
| LightGlu3D (k=0.5, λ=0.05) |              |            |               |            |             |
| LightGlu3D (k=0.5, λ=0.1)  |              |            |               |            |             |



#### Aachen Day&Night

For Aachen, need to save the estimated query pose in .txt file and evaluate on their website: https://www.visuallocalization.net/.

1. MNN:

   ```
   python -m evaluation.pose_estimation \
    --dataset_type aachen \
    --dataset [YOUR_AACHEN_FOLDER](aachen_v1.1) \
    --covisibility_dir [YOUR_AACHEN_OUTPUT_FOLDER]/covisibility \
    --query_dir [YOUR_AACHEN_FOLDER]/CambridgeLandmarks_Colmap_Retriangulated_1024px \
    --sfm_dir [YOUR_AACHEN_OUTPUT_FOLDER]/triangulation \
    --method MNN \
    --outputs evaluation/Aachen_file
   ```

   

2. PR:

   ```
   python -m evaluation.pose_estimation \
    --dataset_type aachen \
    --dataset [YOUR_AACHEN_FOLDER](aachen_v1.1) \
    --covisibility_dir [YOUR_AACHEN_OUTPUT_FOLDER]/covisibility \
    --query_dir [YOUR_AACHEN_FOLDER]/CambridgeLandmarks_Colmap_Retriangulated_1024px \
    --sfm_dir [YOUR_AACHEN_OUTPUT_FOLDER]/triangulation \
    --method PR \
    --outputs evaluation/Aachen_file
   ```

   

3. HLOC:

   ```
   python -m evaluation.pose_estimation \
    --dataset_type aachen \
    --dataset [YOUR_AACHEN_FOLDER](aachen_v1.1) \
    --covisibility_dir [YOUR_AACHEN_OUTPUT_FOLDER]/covisibility \
    --query_dir [YOUR_AACHEN_FOLDER]/CambridgeLandmarks_Colmap_Retriangulated_1024px \
    --sfm_dir [YOUR_AACHEN_OUTPUT_FOLDER]/triangulation \
    --method HLOC \
    --outputs evaluation/Aachen_file
   ```

   

4. LightGlu3D:

   Normally inference:

   ```
   python -m evaluation.pose_estimation \
    --dataset_type aachen \
    --dataset [YOUR_AACHEN_FOLDER](aachen_v1.1) \
    --covisibility_dir [YOUR_AACHEN_OUTPUT_FOLDER]/covisibility \
    --query_dir [YOUR_AACHEN_FOLDER]/CambridgeLandmarks_Colmap_Retriangulated_1024px \
    --sfm_dir [YOUR_AACHEN_OUTPUT_FOLDER]/triangulation \
    --method TRAIN /
    --checkpoint [YOUR_TRAINED_WEIGHTS](.tar) \
    --filter_threshold 0.05
   ```

   

   With dynamic greedy filter threshold strategy in inference:

   ```
   python -m evaluation.pose_estimation \
    --dataset_type aachen \
    --dataset [YOUR_AACHEN_FOLDER](aachen_v1.1) \
    --covisibility_dir [YOUR_AACHEN_OUTPUT_FOLDER]/covisibility \
    --query_dir [YOUR_AACHEN_FOLDER]/CambridgeLandmarks_Colmap_Retriangulated_1024px \
    --sfm_dir [YOUR_AACHEN_OUTPUT_FOLDER]/triangulation \
    --method TRAIN /
    --checkpoint [YOUR_TRAINED_WEIGHTS](.tar) \
     --greedy_or_not True \
    --min_matches 100
   ```

   

Results:

AUC of translation and rotation error:

Each block is in this format: (0.25m, 2◦) / (0.05m, 5◦) / (5.00m, 10◦).

| Method                         | Failed PnP | Day  | Night |
| ------------------------------ | ---------- | ---- | ----- |
| MNN                            |            |      |       |
| PR                             |            |      |       |
| HLOC                           |            |      |       |
| LightGlu3D (k=0.3, match=1000) |            |      |       |
| LightGlu3D (k=0.3, match=800)  |            |      |       |
| LightGlu3D (k=0.3, match=400)  |            |      |       |
| LightGlu3D (k=0.3, λ=0.025)    |            |      |       |
| LightGlu3D (k=0.3, λ=0.05)     |            |      |       |
| LightGlu3D (k=0.3, λ=0.1)      |            |      |       |
| LightGlu3D (k=0.5, match=1000) |            |      |       |
| LightGlu3D (k=0.5, match=800)  |            |      |       |
| LightGlu3D (k=0.5, match=400)  |            |      |       |
| LightGlu3D (k=0.5, λ=0.025)    |            |      |       |
| LightGlu3D (k=0.5, λ=0.05)     |            |      |       |
| LightGlu3D (k=0.5, λ=0.1)      |            |      |       |


## Visualization

### Quantile normalization

For a random query in one scene, visualize how the normalization works on the visible points.

```
python -m visualization.visualize_normalization \
  --covisibility_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/covisibility \
  --query_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/query \
  --sfm_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/triangulation \
  --scene [SCENE_ID]
```


### Matching performance

```
python -m visualization.visualize_matches \
 --dataset [YOUR_MEGADPETH_FOLDER]/Undistorted_SfM \
 --covisibility_dir [YOUR_MEGADEPTH_OUTPUT_FOLDER]/covisibility \
 --query_dir  [YOUR_MEGADEPTH_OUTPUT_FOLDER]/query \
 --sfm_dir  [YOUR_MEGADEPTH_OUTPUT_FOLDER]/triangulation \
 --depth_dir [YOUR_MEGADPETH_FOLDER]/depth_undistorted \
 --scene [SCENE_ID] \
 --method TRAIN \
 --checkpoint [YOUR_TRAINED_WEIGHTS](.tar) \
 --filter_threshold [YOUR_LAMBDA]
```

#### The examples

The matching performance of two examples:

One example in 0015:

One example in 0022:
