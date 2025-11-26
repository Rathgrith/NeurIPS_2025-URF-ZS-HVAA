#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1

dataset_dir="./data/ucf_crime"
llm_model_name="llama3.1-8b"
batch_size=16
threshold=0.05
frame_interval=16
vlm_name="videollama3"
exp_name="WHATEVER_YOU_HAVE_RUN_IN_CAPTION_STEP"
dataset_prior="or criminal"

optimal_scores_dir="${dataset_dir}/scores/${vlm_name}"
refined_scores_dir="${dataset_dir}/refined_scores/${vlm_name}"
suspicious_phrases_json="${dataset_dir}/scores/suspicious_part_phrases.json"
score_window_file="${dataset_dir}/scores/${vlm_name}/highest_lowest_intervals.json"

mkdir -p "${refined_scores_dir}"

echo "Processing index: ${vlm_name}"

root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

context_prompt="How would you rate the scene on a scale from 0 to 1, where 0 is ordinary and 1 indicates suspicious ${dataset_prior} activity? In addition, we have identified certain suspicious or criminal behaviors that may appear in the video. Please consider these carefully when deciding on the final anomaly rating."
format_prompt="Please provide the response as a Python list with exactly one number in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]. Include no extra text. The list should begin with '[' and end with ']'."

captions_dir="${dataset_dir}/captions/${exp_name}"

echo "Output scores directory: ${refined_scores_dir}"

torchrun \
    --nproc_per_node 1 --nnodes 1 --master_port=29500 -m src.refine_with_tag \
    --root_path "${root_path}" \
    --annotationfile_path "${annotationfile_path}" \
    --batch_size "${batch_size}" \
    --frame_interval "${frame_interval}" \
    --captions_dir "${captions_dir}" \
    --context_prompt "${context_prompt}" \
    --format_prompt "${format_prompt}" \
    --output_scores_dir "${refined_scores_dir}" \
    --ckpt_dir ./libs/llama/llama3.1-8b/ \
    --tokenizer_path ./libs/llama/llama3.1-8b/tokenizer.model \
    --suspicious_phrases_json "${suspicious_phrases_json}" \
    --highest_lowest_json "${score_window_file}" \
    --threshold "${threshold}"
# UNSET threshold if you what adaptive margin
cp -n "${optimal_scores_dir}"/*.json "${refined_scores_dir}"/
