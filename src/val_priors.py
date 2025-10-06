import argparse
import os
import pdb
import json
import pickle
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# ==============================
# Hyperparameters for detection
# ==============================
ALPHA = 0.1   # fraction of frames that must be detected for a track to count
BETA = 0.1    # IoU threshold to consider a detection correct
CONF_THRESH = 0.5  # Confidence threshold for final detection metrics

# ---------------------------------
# 0) Initialize Qwen2.5 VL Model
# ---------------------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# ---------------------------------
# 1) Load the JSON for Video-Level Anomalies
# ---------------------------------
with open("./data/ucf_crime/suspicious_part_phrases.json", "r") as f:
    video_anomaly_map = json.load(f)


# ---------------------------------
# 2) Utility Functions
# ---------------------------------
def parse_bbox_confidence_from_text(text):
    """
    Attempt to parse JSON of the form:
        [
          {
            "bbox_2d": [x1, y1, x2, y2],
            "confidence": 0.xx
          }
        ]

    If parsing fails, returns (None, 0.0) by default.
    """
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) > 0:
            first_item = parsed[0]
            if "bbox_2d" in first_item and "confidence" in first_item:
                bbox = first_item["bbox_2d"]
                conf = float(first_item["confidence"])
                if len(bbox) == 4:
                    return list(map(int, bbox)), conf
    except Exception:
        pass

    # Fallback if JSON parse fails:
    print("Could not fully parse JSON. Attempting fallback parse.")
    text_clean = re.sub(r"bbox[\s_]?2d", "", text, flags=re.IGNORECASE)
    matches = re.findall(r"(\d+)", text_clean)
    bbox = None
    if len(matches) >= 4:
        # We just take the first 4 integers found
        bbox = list(map(int, matches[:4]))

    # For confidence, look for something like 0.xx
    conf_match = re.findall(r"(\d*\.\d+)", text)
    conf = 0.0
    if conf_match:
        try:
            conf = float(conf_match[-1])  # last match
        except Exception:
            pass

    return bbox, conf

def compute_iou(boxA, boxB):
    """
    Intersection over Union for [x1,y1,x2,y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def visualize_detection(image, gt_bbox, pred_bbox, title="Detection"):
    """
    Draw ground truth (green) + predicted (red) boxes on the image.
    """
    plt.figure()
    plt.imshow(image)
    ax = plt.gca()

    # Ground truth in green
    gt_rect = plt.Rectangle(
        (gt_bbox[0], gt_bbox[1]),
        gt_bbox[2] - gt_bbox[0],
        gt_bbox[3] - gt_bbox[1],
        fill=False, edgecolor='green', linewidth=2, label='GT'
    )
    ax.add_patch(gt_rect)

    # Prediction in red
    if pred_bbox is not None:
        pred_rect = plt.Rectangle(
            (pred_bbox[0], pred_bbox[1]),
            pred_bbox[2] - pred_bbox[0],
            pred_bbox[3] - pred_bbox[1],
            fill=False, edgecolor='red', linewidth=2, label='Pred'
        )
        ax.add_patch(pred_rect)

    plt.title(title)
    plt.legend()
    plt.savefig("detection_vis.png")  # Overwrites each time
    print("Visualization saved to detection_vis.png")


def compute_filtered_iou(detection_results):
    """
    Compute an overall "filtered" IoU metric based on final_iou (which is set to 0
    if confidence < 0.5). Specifically:
      - Group final_iou by video.
      - For each video, compute average final_iou over its frames.
      - Then compute the mean across videos.

    Return: (filtered_mean_iou, video_avg_iou_dict).
    """
    video_iou = {}
    for det in detection_results:
        vid = det["video"]
        if vid not in video_iou:
            video_iou[vid] = []
        video_iou[vid].append(det["final_iou"])

    video_avg_iou = {}
    for vid, iou_list in video_iou.items():
        if len(iou_list) == 0:
            video_avg_iou[vid] = 0.0
        else:
            video_avg_iou[vid] = sum(iou_list) / len(iou_list)

    # Mean across all videos
    if len(video_avg_iou) == 0:
        filtered_mean_iou = 0.0
    else:
        filtered_mean_iou = sum(video_avg_iou.values()) / len(video_avg_iou)

    return filtered_mean_iou, video_avg_iou

# ---------------------------------
# Single-Phase Qwen2.5 VL Approach
# ---------------------------------
def qwen_single_phase_grounding(image, anomaly_label):
    """
    Ask Qwen2.5-VL to localize suspicious region and return (bbox, confidence).
    """
    lower_label = anomaly_label.strip().lower()
    if not lower_label or "no" in lower_label or anomaly_label == "[]":
        prompt_text = (
            "Analyze this image and identify any suspicious or anomalous region, if present.\n"
            "Return your answer in JSON format, the confidence also reflects the probability of anomaly presence: "
            "[{\"bbox_2d\": [x1, y1, x2, y2], \"confidence\": c}]."
        )
    else:
        prompt_text = (
            f"The video could contain the following anomaly type: '{anomaly_label}'.\n"
            "Localize the suspicious region or individual in this image.\n"
            "Return your answer in JSON format, the confidence also reflects the probability of anomaly presence: "
            "[{\"bbox_2d\": [x1, y1, x2, y2], \"confidence\": c}]."
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    gen_ids = model.generate(**inputs, max_new_tokens=128)
    gen_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
    output_text = processor.batch_decode(
        gen_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"[Single-Phase] Model output:\n{output_text}\n")
    bbox, conf = parse_bbox_confidence_from_text(output_text)
    print(bbox, conf)
    return bbox, conf

def save_bboxes_for_replication(detection_results, output_file="predicted_bboxes_with_conf.json"):
    with open(output_file, "w") as f:
        json.dump(detection_results, f, indent=4)
    print(f"Saved predicted bounding boxes for replication to {output_file}")

# -------------------------------------------
# Dataset & DataLoader
# -------------------------------------------
def anomaly_collate_fn(batch):
    images, coords, paths, video_names = [], [], [], []
    for (img, c, p, v) in batch:
        images.append(img)
        coords.append(c)
        paths.append(p)
        video_names.append(v)
    return images, coords, paths, video_names

class AnomalyDataset(Dataset):
    def __init__(self, annotation_dict, transform=None):
        """
        annotation_dict: {video_name: [ [img_path, x1, y1, x2, y2], ... ]}.
        We'll flatten for iteration.
        """
        self.samples = []
        self.transform = transform
        for video_name, bbox_list in annotation_dict.items():
            for entry in bbox_list:
                img_path, x1, y1, x2, y2 = entry
                self.samples.append((img_path, (x1, y1, x2, y2), video_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, coords, video_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, coords, img_path, video_name

# -------------------------------------------
# Main: Single-Phase + Track/Region Metrics
# -------------------------------------------
if __name__ == "__main__":

    # 1) Load bounding-box annotation dictionary
    with open("Test_annotation_aligned.pkl", "rb") as f:
        annotation_dict = pickle.load(f)
    
    # 2) Create dataset & dataloader
    dataset = AnomalyDataset(annotation_dict)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=anomaly_collate_fn)

    # Structures for storing results
    results_by_video = {}
    detection_results = []  # includes both raw IoU and final IoU
    total_frames = 0
    total_anomalous_regions = 0

    # For final metrics at conf>=0.5
    total_detected_regions = 0
    total_false_positives = 0
    iou_list_for_final = []

    # -------------
    # Loop
    # -------------
    for (images, coords, paths, video_names) in tqdm(data_loader):
        image = images[0]
        gt_bbox = coords[0]
        img_path = paths[0]
        video_name = video_names[0]

        if video_name not in results_by_video:
            results_by_video[video_name] = {
                "frame_count": 0,
                "gt_detected_count": 0,   # for final metrics
                "gt_total_count": 0,
            }

        results_by_video[video_name]["frame_count"] += 1
        results_by_video[video_name]["gt_total_count"] += 1
        total_frames += 1
        total_anomalous_regions += 1

        # Anomaly label
        json_key = video_name
        anomaly_label = video_anomaly_map.get(json_key, "").strip()
        # anomaly_label = video_name
        # Call Qwen => raw box + raw confidence
        pred_bbox, pred_confidence = qwen_single_phase_grounding(image, anomaly_label)

        if pred_bbox is None:
            raw_iou = 0.0
            raw_correct = 0
        else:
            raw_iou = compute_iou(gt_bbox, pred_bbox)
            raw_correct = 1 if raw_iou >= BETA else 0

        # FINAL IoU at conf>=0.5 => for final metrics
        if (pred_bbox is not None) and (pred_confidence >= CONF_THRESH):
            final_iou = raw_iou  # same IoU if we keep the box
            if final_iou >= BETA:
                results_by_video[video_name]["gt_detected_count"] += 1
                total_detected_regions += 1
                final_correct = 1
            else:
                # false positive
                total_false_positives += 1
                final_correct = 0
        else:
            # treat as no detection
            final_iou = 0.0
            final_correct = 0

        iou_list_for_final.append(final_iou)
        detection_results.append({
            "video": video_name,
            "gt_bbox": gt_bbox,
            "pred_bbox": pred_bbox,
            "confidence": float(pred_confidence),
            "raw_iou": raw_iou,           
            "raw_correct": raw_correct,   
            "final_iou": final_iou,      
            "final_correct": final_correct
        })


    # -------------------
    # Compute final metrics @ conf>=0.5
    # -------------------
    for vid, stats in results_by_video.items():
        gt_total = stats["gt_total_count"]
        if gt_total > 0:
            frac_detected = stats["gt_detected_count"] / float(gt_total)
        else:
            frac_detected = 0.0

    mean_iou_final = sum(iou_list_for_final) / len(iou_list_for_final) if iou_list_for_final else 0.0

    print("\n=========== Final Metrics (Fixed conf>=0.5) ===========")
    print(f"TIoU (conf>=0.5): {mean_iou_final:.4f}")

    # Save JSON for replication
    save_bboxes_for_replication(detection_results, output_file="predicted_bboxes_with_probs_priors.json")
