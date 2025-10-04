# ucf-crime dataset directory
#!/bin/bash

dataset_dir="./data/ucf_crime"
videos_dir="${dataset_dir}/videos"
frames_dir="${dataset_dir}/frames"
annotations_file="${dataset_dir}/annotations/test.txt"
ground_truth_file="${dataset_dir}/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"

# Create frames directory if it doesn't exist
mkdir -p "$frames_dir"
# Extract frames using the Python script
python ./src/extract_frames.py \
    --videos_dir "$videos_dir" \
    --frames_dir "$frames_dir" \
    --annotations_file "$annotations_file" \
    --ground_truth_file "$ground_truth_file"

# xd-violence dataset directory
dataset_dir="./data/xd_violence"

# Set paths
videos_dir="${dataset_dir}/videos"
frames_dir="${dataset_dir}/frames"
# adjust file names accordingly, as the original distributor of them use different namings
annotations_file="${dataset_dir}/annotations/test.txt"
ground_truth_file="${dataset_dir}/annotations/temporal_anomaly_annotation_for_testing_videos.txt"

# ls "$videos_dir"
python ./src/extract_frames.py \
    --videos_dir "$videos_dir" \
    --frames_dir "$frames_dir" \
    --annotations_file "$annotations_file" \
    --ground_truth_file "$ground_truth_file"

# xd-violence dataset directory
dataset_dir="./data/UBNormal"

# Set paths
videos_dir="${dataset_dir}/videos"
frames_dir="${dataset_dir}/frames"
annotations_file="${dataset_dir}/annotations/test.txt"
ground_truth_file="${dataset_dir}/annotations/temporal.txt"

# ls "$videos_dir"
python ./src/extract_frames.py \
    --videos_dir "$videos_dir" \
    --frames_dir "$frames_dir" \
    --annotations_file "$annotations_file" \
    --ground_truth_file "$ground_truth_file"

# xd-violence dataset directory
dataset_dir="./data/MSAD"

# Set paths
videos_dir="${dataset_dir}/videos"
frames_dir="${dataset_dir}/frames"
annotations_file="${dataset_dir}/annotations/test.txt"
ground_truth_file="${dataset_dir}/annotations/msad_anomaly_index.txt"

# ls "$videos_dir"
python ./src/extract_frames.py \
    --videos_dir "$videos_dir" \
    --frames_dir "$frames_dir" \
    --annotations_file "$annotations_file" \
    --ground_truth_file "$ground_truth_file"

