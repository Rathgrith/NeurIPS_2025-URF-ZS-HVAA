import argparse
import os
from pathlib import Path
from tqdm import tqdm
import cv2

def extract_frames(video_path, frames_dir):
    video_name = video_path.split("/")[-1].replace(".mp4", "")
    video_frames_dir = os.path.join(frames_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(video_frames_dir, f"{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {video_frames_dir}")
    return video_name, frame_count


def main(videos_dir, frames_dir, annotations_file, ground_truth_file):
    with open(ground_truth_file, "r") as f:
        ground_truth_videos = set(line.split()[0] for line in f)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
    # disable writing as the files are already there, uncomment for new dataset need to write test.txt.
    # with open(annotations_file, "w") as f:
    for video_file in tqdm(os.listdir(videos_dir)):
        video_name = video_file.split("/")[-1].replace(".avi", "").replace(".mp4", "")
        if video_name in ground_truth_videos:
            video_path = os.path.join(videos_dir, video_file)
            video_name, num_frames = extract_frames(video_path, frames_dir)
            # f.write(f"{video_name} 0 {num_frames - 1} 0\n")
        else:
            print(video_file, ground_truth_videos)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="Directory path to the videos.",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory path to the frames.",
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        required=True,
        help="Path to the annotations file.",
    )
    parser.add_argument(
        "--ground_truth_file",
        type=str,
        required=True,
        help="Path to the ground truth file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.videos_dir, args.frames_dir, args.annotations_file, args.ground_truth_file)