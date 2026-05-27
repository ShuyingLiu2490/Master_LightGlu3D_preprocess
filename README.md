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

mamba create --name matchenv python=3.10
mamba activate matchenv

Install the following repos:

1. LightGlue

   git clone https://github.com/cvg/LightGlue.git
   cd LightGlue
   python -m pip install -e .

2. Gluefactory

   git clone https://github.com/cvg/glue-factory.git
   cd glue-factory
   python3 -m pip install -e .

3. HLOC

   git clone https://github.com/cvg/Hierarchical-Localization.git
   cd Hierarchical-Localization
   python -m pip install -e .

   git submodule update --init --recursive

   

## Install the datasets

1. Megadepth
2. Cambridge Landmarks
3. Aachen Day&Night v1.1



## File Structure

```
├──preprocess_Megadepth
│   ├──extract_query_sets.py # Split dataset into query and reference
│   ├──triangulation_gpu_steps.py # Feature extraction and matching preparson for triangulation
│   ├──triangulation_cpu_steps.py # Triangulation
│   ├──covisibility_search_pipe.py # Find visible 3D points for each query based on covisibility expansion
│   ├──precompute_features.py # Calculate averaged 3D features
│   └──generate_gt_pairs_by_scene.py # GT calculation functions
│ 
├──preprocess_extra # extra preprocess code for Cambridge and Aachen
│   ├──triangulation_aachen.py # Triangulation Aachen in original SIFT coordinate
│   ├──covisibility_search_pipe_aachen.py # Covisibility search on Aachen
│   └──precompute_features_aachen.py # Calculated averaged 3D features on Aachen
│ 
├──baselines_and_trained_matcher
│   ├──mnn_baseline.py # Mutual Nearest Neighbour (MNN) matching baseline, load from gluefactory
│   ├──pr_lg_baseline.py # Projection Reference LightGlue (PR) matching baseline
│   ├──lightglu3d_bicross.py # Final LightGlu3D matcher benchmark
│   ├──trained_matcher.py # Use the trained matcher and dynamic strategy
│   └──network_weights # Put the trained weights here
│
├──evaluation
│   ├──cambridge_selected.txt # Scene list for Cambridge Landmarks
│   ├──inference.py # Match performance for all matchers on all datasets
│   └──pose_estimation.py # Absolute pose estimation for all matchers on all datasets
│
├──visualization
│   ├──visualize_gt.py # Visualization of the soft threshold effect
│   ├──visualize_normalization.py # Visualization of 3D normalization
│   ├──visualize_matches.py # Visualziation of the matching
│   └──visualize_no_match.py # Visualization the no matching case
```



## Preprocess

First is 

### Split uery and reference

In Megadepth, we have train.txt, valid.txt, and test.txt. Also, can set small scenes try.

Some arguments could try: sample_ratio: how many 3D points are considered from the original SIFT 3D model, query_ratio: how many images could be query.

```
python -m preprocess_Megadepth.extract_query_sets \
 --outputs [YOUR_OUTPUT_FOLDER]/query \
 --scene [SCENE_NAME]
```



### Triangulation

Triangulation based on Superpoint feature and LightGlue matching.  This part the code is seperated in GPU step and CPU step.

GPU part is the feature matching. CPU part is the triangulation.

Some arguments could try in GPU part: min_overlap and max_overlap: the score threshold from the overlap matrix in scene_info.

```
python -m preprocess_Megadepth.triangulation_gpu_steps \
 --dataset [YOUR_MEGADPETH_FOLDER]/Undistorted_SfM \
 --outputs [YOUR_OUTPUT_FOLDER]/triangulation \
 --query_dir [YOUR_OUTPUT_FOLDER]/query \
 --scene_list [SCENES_TXT_FILE]
 
python -m preprocess_Megadepth.triangulation_cpu_steps \
 --dataset [YOUR_MEGADPETH_FOLDER]/Undistorted_SfM \
 --outputs [YOUR_OUTPUT_FOLDER]/triangulation \
 --html_save_dir [YOUR_OUTPUT_FOLDER]/saved_html_visual \
 --scene_list [SCENES_TXT_FILE]
```



### Covisibility search

Find the visible 3D points for each query.

Some arguments could try: pruning: pruning factor for covisible overlap.

```
python -m preprocess_Megadepth.covisibility_search_pipe \
 --dataset [YOUR_MEGADPETH_FOLDER]/Undistorted_SfM \
 --outputs [YOUR_OUTPUT_FOLDER]/covisibility \
 --sfm_dir [YOUR_OUTPUT_FOLDER]/triangulation \
 --query_list [YOUR_OUTPUT_FOLDER]/query \
 --scene_list [SCENES_TXT_FILE]
```



### Calculate 3D features

Average the 2D features from track of each 3D points to get the 3D features.

```
python -m preprocess_Megadepth.precompute_features \
 --dataset [YOUR_MEGADPETH_FOLDER]/Undistorted_SfM \
 --outputs [YOUR_OUTPUT_FOLDER]/covisibility \
 --sfm_dir [YOUR_OUTPUT_FOLDER]/triangulation \
 --scene_list [SCENES_TXT_FILE]
```



### Ground Truth Calculation

The ground truth calculation functions could be found in `preprocess_Megadepth.generate_gt_pairs_by_scene`, including the calculation with and without the ignore label. Both consider reprojection error and depth error.



### Preprocess for Aachen

In Aachen, it has already separated the query and reference. So skip that step.

Start from the triangulation. (The dataset we put is not in a good structure, so we add image_dir here.)

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



## Baselines

