#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

dataset_dir="./data/ucf_crime"
llm_model_name="llama3.1-8b"

batch_size=32
frame_interval=16
vlm_name="video_llama3"
seed=1

echo "Processing index: ${vlm_name}"

root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"
context_prompt="How would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities or potentially criminal activities?"
format_prompt="Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with  ']'."

output_scores_dir="${dataset_dir}/scores/${vlm_name}"
captions_dir="${dataset_dir}/captions/${vlm_name}_json_results"

mkdir -p "${output_scores_dir}"

echo "Writing scores to: ${output_scores_dir}"
echo "Using captions from: ${captions_dir}"

torchrun \
    --nproc_per_node 1 --nnodes 1 --master_port=29501 -m src.llm_anomaly_scorer \
    --root_path "${root_path}" \
    --annotationfile_path "${annotationfile_path}" \
    --batch_size "${batch_size}" \
    --frame_interval "${frame_interval}" \
    --captions_dir "${captions_dir}" \
    --context_prompt "${context_prompt}" \
    --format_prompt "${format_prompt}" \
    --output_scores_dir "${output_scores_dir}" \
    --ckpt_dir ./libs/llama/llama3.1-8b/ \
    --tokenizer_path ./libs/llama/llama3.1-8b/tokenizer.model \
    --seed "${seed}"
