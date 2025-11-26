import os
import torch
import json
import cv2
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
import numpy as np
import random

SEED = 3306
torch.manual_seed(SEED)
np.random.seed(SEED)

random.seed(SEED)

if torch.cuda.is_available():
    device = "cuda:0"
    torch.cuda.manual_seed_all(SEED)

def load_model():
    model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

@torch.inference_mode()
def infer(model, processor, conversation):
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # Move tensors to GPU if available
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and torch.cuda.is_available():
            inputs[k] = v.to(device)

    # If pixel_values exist, ensure they're the correct dtype
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

    output_ids = model.generate(**inputs, max_new_tokens=1024)
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]

def get_video_fps_and_duration(video_path):
    """
    Returns (fps, duration_in_seconds) of the video at `video_path`.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps == 0:
        return (0, 0)
    duration = frame_count / fps
    return (fps, duration)

def process_suspicious_interval(model, processor, video_path, start_frame, end_frame):
    """
    Summarizes the suspicious interval (start_frame to end_frame) in the video.
    Converts frames to seconds using actual FPS. Then passes that to the model with a
    prompt indicating this is the suspicious segment, asking for possible crime behaviors.
    """
    fps, _ = get_video_fps_and_duration(video_path)
    if fps <= 0:
        print(f"Warning: FPS=0 for {video_path}, skipping.")
        return ""

    # Convert frame indexes to seconds
    start_sec = start_frame / fps
    end_sec = end_frame / fps

    # Ensure end_sec isn't beyond the actual video length
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    video_duration_sec = total_frames / fps
    end_sec = min(end_sec, video_duration_sec)
    
    max_frames = 180
    
    conversation = [
        {
            "role": "system",
            "content": "You are an AI assistant analyzing a suspicious segment of a video."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": video_path,
                        "fps": 18,
                        "start_time": start_sec,
                        "end_time": end_sec,
                        "max_frames": max_frames
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Analyze the video interval to identify any possible suspicious behaviors. "
                        "Return your answer strictly as a Python-style list of phrases that could briefly describe "
                        "the suspicious scene splited by commas."
                        "No additional commentary or text, return only the list."
                    )
                }
            ]
        }
    ]

    response = infer(model, processor, conversation)
    print(response)
    torch.cuda.empty_cache()
    return response.strip()


def main():
    # Paths
    folder_path="../ucf_crime/videos/"
    index_file = "../ucf_crime/annotations/test.txt"
    intervals_file = "highest_lowest_intervals.json"
    output_file = "suspicious_part_phrases.json"

    # Load the model and processor
    model, processor = load_model()

    # Load intervals => e.g. { "Abuse028_x264": { "interval": [48, 96], "average_score": 0.2 }, ... }
    with open(intervals_file, "r") as f:
        suspicious_intervals = json.load(f)

    # Prepare to store final results
    results = {}

    # We read the index file lines. For each line, we have a video filename (w/o extension).
    with open(index_file, "r") as f:
        video_files = [line.split()[0] for line in f.readlines()]

    for filename in tqdm(video_files, desc="Processing suspicious intervals"):
        # Example: If line is "Abuse/Abuse028_x264", then the "base" might be "Abuse028_x264"
        # But your intervals are keyed as "Abuse028_x264" (no folder, no .mp4).
        # We'll strip the folder if present, plus .mp4 if present, to match the intervals dict keys.
        base_name = filename.split("/")[-1]  # e.g. "Abuse028_x264"

        # If that base_name is in your intervals file, get the suspicious [start_frame, end_frame].
        if base_name not in suspicious_intervals:
            print(f"No suspicious interval found for {base_name}, skipping.")
            continue

        interval_info = suspicious_intervals[base_name]
        start_frame, end_frame = interval_info["highest_interval"]  # e.g. [48, 96]

        video_path = os.path.join(folder_path, base_name + ".mp4")
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} not found.")
            continue

        # Summarize the suspicious interval
        summary = process_suspicious_interval(
            model, processor, video_path,
            start_frame=start_frame,
            end_frame=end_frame
        )

        # Save it
        results[filename] = summary
        # {
        #     # "start_frame": start_frame,
        #     # "end_frame": end_frame,
        #     # "description": summary
        # }

    # Write all suspicious-part summaries to a JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Done. Saved suspicious part summaries to {output_file}")

if __name__ == "__main__":
    main()
