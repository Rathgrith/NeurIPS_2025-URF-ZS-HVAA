#!/bin/bash
export OMP_NUM_THREADS=8

dataset_dir="./data/UBNormal"
llm_model_name="llama3.1-8b"
vlm_name="videollama3_7B"
frame_interval=16
video_fps=30

root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

scores_dir="${dataset_dir}/refined_scores/${vlm_name}"
captions_dir="${dataset_dir}/captions/${vlm_name}"
output_dir="${scores_dir}/metrics"

echo "Evaluating scores from: ${scores_dir}"
mkdir -p "${output_dir}"

python -m src.eval \
    --root_path "${root_path}" \
    --annotationfile_path "${annotationfile_path}" \
    --scores_dir "${scores_dir}" \
    --captions_dir "${captions_dir}" \
    --output_dir "${output_dir}" \
    --frame_interval "${frame_interval}" \
    --temporal_annotation_file "${dataset_dir}/annotations/temporal.txt" \
    --video_fps "${video_fps}"
