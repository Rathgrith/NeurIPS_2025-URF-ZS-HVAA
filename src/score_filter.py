import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_scores(scores, sigma=10):
    frames_int = sorted(map(int, scores.keys()))
    if not frames_int:
        return {}
    arr = np.zeros(frames_int[-1] + 1, dtype=np.float32)
    for f in frames_int:
        arr[f] = scores[str(f)]
    smoothed = gaussian_filter1d(arr, sigma=sigma)
    return {str(f): float(smoothed[f]) for f in frames_int}


def find_extreme_intervals(scores):
    frames_int = sorted(map(int, scores.keys()))
    if not frames_int:
        return (0, 0, 0.0, 0, 0, 0.0)

    max_frame = frames_int[-1]
    window_size = max(max_frame // 10, 300)

    best_avg, worst_avg = float('-inf'), float('inf')
    best_s = best_e = worst_s = worst_e = 0

    n = len(frames_int)
    for i in range(n):
        s = frames_int[i]
        e = s + window_size
        window = [scores[str(fr)] for fr in frames_int[i:n] if fr < e]
        if not window:
            continue
        avg = sum(window) / len(window)
        if avg > best_avg:
            best_avg, best_s, best_e = avg, s, e
        if avg < worst_avg:
            worst_avg, worst_s, worst_e = avg, s, e

    return best_s, best_e, best_avg, worst_s, worst_e, worst_avg


def pick_top_frames(scores, k=16, min_distance=30):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    chosen = []
    for frame_str, _ in ranked:
        idx = int(frame_str)
        if all(abs(idx - c) >= min_distance for c in chosen):
            chosen.append(idx)
            if len(chosen) == k:
                break
    return sorted(chosen)



def group_stats(all_scores, best_avg, worst_avg):
    'Return std_dev, avg_high_group, avg_low_group, gap.'
    vals = np.array(list(all_scores.values()), dtype=np.float32)
    std_dev = float(np.std(vals))

    high_group = vals[np.abs(vals - best_avg) <= np.abs(vals - worst_avg)]
    low_group = vals[np.abs(vals - best_avg) > np.abs(vals - worst_avg)]

    if high_group.size == 0:
        high_group = vals
    if low_group.size == 0:
        low_group = vals

    avg_high = float(high_group.mean())
    avg_low = float(low_group.mean())
    gap = avg_high - avg_low
    return std_dev, avg_high, avg_low, gap


def main():
    dataset_dir = Path('./data/ucf_crime')
    vlm_name = 'videollama3_8B'
    score_dir = dataset_dir / 'scores' / vlm_name
    intervals_output_json = score_dir / 'highest_lowest_intervals.json'

    if not score_dir.exists():
        raise FileNotFoundError(f'Score directory not found: {score_dir}')

    intervals_results = {}
    excluded_files = {intervals_output_json.name, 'suspicious_part_phrases.json'}

    for json_file in sorted(score_dir.glob('*.json')):
        if json_file.name in excluded_files:
            continue

        with json_file.open('r') as f:
            raw_scores = json.load(f)
        if not raw_scores:
            continue

        base_name = json_file.stem
        best_s, best_e, best_avg, worst_s, worst_e, worst_avg = find_extreme_intervals(raw_scores)
        std, avg_high, avg_low, gap = group_stats(raw_scores, best_avg, worst_avg)

        smoothed = smooth_scores(raw_scores)
        top_frames = pick_top_frames(smoothed, k=16, min_distance=30)

        intervals_results[base_name] = {
            'highest_interval': [best_s, best_e],
            'highest_avg_score': round(best_avg, 3),
            'lowest_interval': [worst_s, worst_e],
            'lowest_avg_score': round(worst_avg, 3),
            'std': round(std, 5),
            'avg_high_group': round(avg_high, 3),
            'avg_low_group': round(avg_low, 3),
            'gap_high_low': round(gap, 3),
            'top_scored_frames': top_frames,
        }

    with intervals_output_json.open('w') as f:
        json.dump(intervals_results, f, indent=4)
    print(f'Saved highest/lowest intervals + stats to {intervals_output_json}')


if __name__ == '__main__':
    main()
