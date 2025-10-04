import json
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


def load_model():
    model_path = 'DAMO-NLP-SG/VideoLLaMA3-7B'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation='flash_attention_2',
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


@torch.inference_mode()
def infer(model, processor, conversation):
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors='pt',
    )

    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor) and torch.cuda.is_available():
            inputs[key] = value.cuda()

    if 'pixel_values' in inputs:
        target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        inputs['pixel_values'] = inputs['pixel_values'].to(target_dtype)

    output_ids = model.generate(**inputs, max_new_tokens=1024)
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]


def get_video_fps_and_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps == 0:
        return (0, 0)
    duration = frame_count / fps
    return (fps, duration)


def process_suspicious_interval(model, processor, video_path, start_frame, end_frame):
    fps, _ = get_video_fps_and_duration(video_path)
    if fps <= 0:
        print(f'Warning: FPS=0 for {video_path}, skipping.')
        return ''

    start_sec = start_frame / fps
    end_sec = end_frame / fps

    cap = cv2.VideoCapture(str(video_path))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    video_duration_sec = total_frames / fps if fps else 0
    end_sec = min(end_sec, video_duration_sec)

    if end_sec <= start_sec:
        print(f'Warning: Invalid interval for {video_path}: {start_frame}-{end_frame}')
        return ''

    max_frames = 180
    conversation = [
        {
            'role': 'system',
            'content': 'You are an AI assistant analyzing part of a video.',
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'video',
                    'video': {
                        'video_path': str(video_path),
                        'fps': max(1, round(fps)),
                        'start_time': start_sec,
                        'end_time': end_sec,
                        'max_frames': max_frames,
                    },
                },
                {
                    'type': 'text',
                    'text': (
                        'Analyze the interval to identify any possible suspicious or criminal behaviors. '
                        'Describe what is happening in this interval in one concise phrase. '
                        'Provide only that phrase in your response.'
                    ),
                },
            ],
        },
    ]

    response = infer(model, processor, conversation)
    return response.strip()


def main():
    dataset_dir = Path('./data/ucf_crime')
    vlm_name = 'videollama3_8B'
    video_folder_path = dataset_dir / 'videos'
    intervals_file = dataset_dir / 'scores' / vlm_name / 'highest_lowest_intervals.json'
    output_file = dataset_dir / 'scores' / vlm_name / 'suspicious_part_phrases.json'
    index_file = dataset_dir / 'annotations' / 'test.txt'

    if not intervals_file.exists():
        raise FileNotFoundError(f'Interval file not found: {intervals_file}')
    if not index_file.exists():
        raise FileNotFoundError(f'Index file not found: {index_file}')

    model, processor = load_model()

    with intervals_file.open('r') as f:
        suspicious_intervals = json.load(f)

    results = {}

    with index_file.open('r') as f:
        video_files = [line.split()[0] for line in f.readlines() if line.strip()]

    for filename in tqdm(video_files, desc='Processing suspicious intervals'):
        base_name = filename.split('/')[-1].replace(".avi", "").replace(".mp4", "")

        if base_name not in suspicious_intervals:
            print(f'No suspicious interval found for {base_name}, skipping.')
            continue

        interval_info = suspicious_intervals[base_name]
        start_frame, end_frame = interval_info['highest_interval']
        # for most datasets, ends with .mp4
        video_path = video_folder_path / f'{base_name}.mp4'
        if not video_path.exists():
            print(f'Warning: {video_path} not found.')
            continue

        summary = process_suspicious_interval(
            model,
            processor,
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        results[base_name] = summary

    with output_file.open('w') as f:
        json.dump(results, f, indent=4)

    print(f'Done. Saved suspicious part summaries to {output_file}')


if __name__ == '__main__':
    main()
