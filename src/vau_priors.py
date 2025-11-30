import os
import torch
import json
import cv2
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

def load_model():
    """
    Load VideoLLaMA3-7B model and processor.
    """
    model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    float_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=float_dtype,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

@torch.inference_mode()
def infer(model, processor, conversation, temperature=0.1):
    """
    Run inference on a single conversation using the model and processor.
    """
    if not conversation:
        print("Warning: Empty conversation input, skipping inference.")
        return ""
    
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    device = next(model.parameters()).device
    float_dtype = next(model.parameters()).dtype
    
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if k == "pixel_values":
                inputs[k] = v.to(device, dtype=float_dtype)
            else:
                inputs[k] = v.to(device)
    
    output_ids = model.generate(**inputs, max_new_tokens=1024, temperature=temperature)
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]

def get_video_length_and_framecount(video_path):
    """
    Returns the duration (in seconds), frames per second, and frame count for a video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration, fps, frame_count

def load_anomaly_data(highest_lowest_json, suspicious_phrases_json):
    """
    Load anomaly probability and suspicious phrases from JSON files.
    """
    with open(highest_lowest_json, 'r') as f:
        highest_lowest_data = json.load(f)
    
    with open(suspicious_phrases_json, 'r') as f:
        suspicious_phrases_data = json.load(f)
    phrases_mapping = {}
    for key, phrases in suspicious_phrases_data.items():
        phrases_mapping[key.split("/")[-1]] = phrases
    anomaly_data = {}
    
    for video_name, data in highest_lowest_data.items():
        # Extract base name by removing _x264 
        base_name = video_name
        probability = data["highest_avg_score"]
        intervals = data["highest_interval"][0] + data["highest_interval"][1]/2
        phrases = phrases_mapping.get(video_name, "No suspicious activity description available")
        
        anomaly_data[video_name] = {
            'probability': probability,
            'suspicious_phrases': phrases,
            'intervals': intervals
        }
    
    return anomaly_data

def process_video(video_path, video_name, model, processor, anomaly_data):
    """
    Process one video with VideoLLaMA3.
    
    The function samples 16 frames (uniformly across the video) and creates a conversation
    using either an enhanced prompt (for high probability anomalies) or vanilla prompt (for low probability).
    """


    # Get video properties
    total_duration, orig_fps, total_frames = get_video_length_and_framecount(video_path)
    if total_duration == 0:
        print(f"Error: Unable to determine duration for {video_path}. Skipping.")
        return None

    # Get anomaly information for this video
    video_key = video_name
    anomaly_info = anomaly_data.get(video_key, {})
    probability = anomaly_info.get('probability', 0)
    # print(anomaly_info, probability)
    suspicious_phrases = anomaly_info.get('suspicious_phrases', "No suspicious activity description available")
    
    # Format the probability for display
    probability_percent = round(probability * 100, 1) if probability else 0
    # Determine which prompt to use based on anomaly probability
    if probability > 0.5 and suspicious_phrases and "no " not in suspicious_phrases:  # Use enhanced prompt for high probability anomalies
        # print(video_path)
        bbox_path = video_path.replace("/videos/", "/videos_with_bboxes/")
        # print(bbox_path)
        if os.path.exists(bbox_path):
            print(f"[info] Using bbox‐overlayed video for {video_name}: {bbox_path}")
            video_path = bbox_path
            print("-----Using bbox overlaid videos.")
        else:
            print(f"[info] No bbox video found for {video_name}, using original: {video_path}")
       
        conversation = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant analyzing a video. "
                    "For better anomaly detection and description in detail, a preliminary analysis suggests that the suspicious activity could be related to [{suspicious_phrases}]. "
                    "Use these information to guide your anomaly detection analysis."
                ).format(probability_percent=probability_percent, suspicious_phrases=suspicious_phrases)
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "start_time": 0,
                            "end_time": total_duration,
                            "max_frames": 16
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Please analyze the video for any anomaly activities in detail. "
                            "If there is any anomaly, describe the anomaly activities present in the video in detail. After description, analyze why it is an anomaly without timestamps."
                            "If no anomalies are found, state that the video appears normal and then describe the scene in detail."
                        ).format(probability_percent=probability_percent, suspicious_phrases=suspicious_phrases)
                    }
                ]
            }
        ]
    else:  # Use vanilla prompt for low probability or no anomaly data
        print("-------Unclear about anomaly type.")
        conversation = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant analyzing a video. "
                    "For better anomaly detection and description in detail, a preliminary analysis suggests that the suspicious activity could be related to {suspicious_phrases}. "
                    "Use these information to guide your anomaly detection analysis."
                    ).format(probability_percent=probability_percent, suspicious_phrases=suspicious_phrases)
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "start_time": 0,
                            "end_time": total_duration,
                            "max_frames": 16
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Please analyze the video for any anomaly activities in detail. "
                            "If there is any anomaly, describe the anomaly activities present in the video in detail. After description, analyze why it is an anomaly without timestamps."
                            "If no anomalies are found, state that the video appears normal and then describe the scene in detail."
                        )
                    }
                ]
            }
        ]
    # Call the model to get the inference (textual response)
    response = infer(model, processor, conversation)
    print(f"Processed {video_name}: {response}")
    return response

def process_videos_from_index(index_file, folder_path, model, processor, output_json, anomaly_data):
    """
    Reads video filenames from an index file, processes each video,
    and saves all model responses in one JSON file.
    """
    if not os.path.exists(index_file):
        print(f"Error: Index file {index_file} not found.")
        return

    responses = {}
    with open(index_file, "r") as f:
        video_files = [line.split()[0] for line in f.readlines() if line.strip()]

    for filename in tqdm(video_files[90:], desc="Processing videos"):
        # Use the basename (without extension) as video identifier.
        base_name = filename.split('/')[-1]
        base_name = base_name.replace(".mp4", "")
        video_path = os.path.join(folder_path, base_name + ".mp4")
        response = process_video(video_path, base_name, model, processor, anomaly_data)
        if response is not None:
            responses[base_name] = response

    # Save all responses in one JSON file
    with open(output_json, "w") as f:
        json.dump(responses, f, indent=4)
    print(f"Processing complete. Responses saved in {output_json}")

if __name__ == "__main__":
    # Define folder paths and index file.
    folder_path = "./data/ucf_crime/videos/"
    index_file = "./data/ucf_crime/annotations/test.txt"
    output_json = "responses_video_llama3_16_InterTC_replication_full_bbox0.5.json"
    # Load anomaly data
    highest_lowest_json = "highest_lowest_intervals_ucf_final.json"
    suspicious_phrases_json = "suspicious_part_phrases.json"
    # uncomment for xd_violence
    anomaly_data = load_anomaly_data(highest_lowest_json, suspicious_phrases_json)

    model, processor = load_model()
    process_videos_from_index(index_file, folder_path, model, processor, output_json, anomaly_data)
    print("Processing complete. Responses saved in", output_json)
