import os
import json
import re
import cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

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
# 1) Load JSONs
# ---------------------------------
with open("./data/ucf_crime/scores/suspicious_part_phrases.json", "r") as f:
    video_anomaly_map = json.load(f)

with open("./data/ucf_crime/scores/highest_lowest_intervals.json", "r") as f:
    interval_map = json.load(f)

# ---------------------------------
# 2) Helpers
# ---------------------------------
def parse_bbox_confidence_from_text(text):
    try:
        parsed = json.loads(text)
        item = parsed[0]
        return list(map(int, item["bbox_2d"])), float(item["confidence"])
    except Exception:
        # Fallback regex parse (as before)…
        # ...omitted for brevity...
        return None, 0.0

def qwen_single_phase_grounding(image, anomaly_label):
    lower_label = anomaly_label.strip().lower()
    if not lower_label or "no" in lower_label or anomaly_label == "[]":
        prompt = (
            "Analyze this image and identify any suspicious or anomalous region, if present.\n"
            "Return JSON: [{\"bbox_2d\": [x1, y1, x2, y2], \"confidence\": c}]."
        )
    else:
        prompt = (
            f"The video could contain anomaly: '{anomaly_label}'.\n"
            "Localize it and return JSON: "
            "[{\"bbox_2d\": [x1, y1, x2, y2], \"confidence\": c}]."
        )
    messages = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
    gen = model.generate(**inputs, max_new_tokens=128)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return parse_bbox_confidence_from_text(out_text)

def get_sampled_frames(interval, step=16):
    start, end = interval
    return list(range(start, end+1, step))

def overlay_bboxes_on_video(video_path, detections, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # map frame_idx → (bbox, conf)
    det_map = {d["frame_idx"]: (d["pred_bbox"], d["confidence"]) for d in detections}

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in det_map:
            bbox, conf = det_map[idx]
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        out.write(frame)
        idx += 1

    cap.release()
    out.release()
    print(f"Saved processed video to {output_path}")

# ---------------------------------
# 3) Main
# ---------------------------------
if __name__ == "__main__":
    detection_results = []
    video_root = os.path.join("..", "ucf_crime", "frames")
    raw_video_root = os.path.join("..", "ucf_crime", "videos")
    out_video_root = os.path.join("..", "ucf_crime", "videos_with_bboxes")

    for video_name, info in interval_map.items():
        interval = info["highest_interval"]
        frames_to_run = get_sampled_frames(interval, step=16)

        anomaly_key = video_name
        anomaly_label = video_anomaly_map.get(anomaly_key, "").strip()

        # run detection on sampled frames
        for idx in frames_to_run:
            img_file = os.path.join(video_root, video_name, f"{idx:06d}.jpg")
            image = Image.open(img_file).convert("RGB")
            bbox, conf = qwen_single_phase_grounding(image, anomaly_label)
            detection_results.append({
                "video": video_name,
                "frame_idx": idx,
                "pred_bbox": bbox,
                "confidence": float(conf),
            })

        # now overlay onto the raw video
        in_vid = os.path.join(raw_video_root, f"{video_name}.mp4")
        out_vid = os.path.join(out_video_root, f"{video_name}_bbox.mp4")
        this_video_dets = [d for d in detection_results if d["video"] == video_name]
        overlay_bboxes_on_video(in_vid, this_video_dets, out_vid)
