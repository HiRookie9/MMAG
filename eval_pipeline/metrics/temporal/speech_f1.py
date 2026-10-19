import json
import sys
import argparse
from collections import defaultdict

def parse_timestamp(ts_str):
    """Parse a ``start-end`` string into a float pair."""
    start, end = ts_str.split('-')
    return float(start), float(end)

def compute_frame_f1(ref_segments, pred_segments, resolution=0.04):
    """
    Compute frame-level F1 from reference and predicted segments.
    """
    # Use the largest segment end as the evaluation duration.
    max_time = 0.0
    for s, e in ref_segments + pred_segments:
        if e > max_time:
            max_time = e
    if max_time == 0:
        # Empty reference and prediction are treated as a perfect match.
        return 1.0

    n_frames = int(max_time / resolution) + 1
    ref_binary = [0] * n_frames
    pred_binary = [0] * n_frames

    # Mark a frame when at least half of it overlaps a segment.
    half_res = resolution * 0.5
    for i in range(n_frames):
        frame_start = i * resolution
        frame_end = frame_start + resolution
        # Compute reference overlap.
        ref_overlap = 0.0
        for s, e in ref_segments:
            overlap = max(0.0, min(frame_end, e) - max(frame_start, s))
            ref_overlap += overlap
        if ref_overlap >= half_res:
            ref_binary[i] = 1

        # Compute prediction overlap.
        pred_overlap = 0.0
        for s, e in pred_segments:
            overlap = max(0.0, min(frame_end, e) - max(frame_start, s))
            pred_overlap += overlap
        if pred_overlap >= half_res:
            pred_binary[i] = 1

    # Compute TP, FP, and FN.
    tp = sum(1 for r, p in zip(ref_binary, pred_binary) if r and p)
    fp = sum(1 for r, p in zip(ref_binary, pred_binary) if not r and p)
    fn = sum(1 for r, p in zip(ref_binary, pred_binary) if r and not p)

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def load_jsonl(filepath):
    """Read a JSONL file into a list."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def main(pred_file, ref_file, output_file):
    # 1. Load predictions.
    pred_data = load_jsonl(pred_file)
    pred_dict = defaultdict(list)
    for item in pred_data:
        audio_id = item['id']
        for seg in item['segments']:
            pred_dict[audio_id].append((seg['start'], seg['end']))

    # 2. Load references.
    ref_data = load_jsonl(ref_file)
    ref_dict = defaultdict(list)
    for item in ref_data:
        audio_id = item['id']
        start, end = parse_timestamp(item['timestamp'])
        ref_dict[audio_id].append((start, end))

    # 3. Evaluate the union of IDs.
    all_ids = sorted(set(ref_dict.keys()) | set(pred_dict.keys()))

    results = []
    total_f1 = 0.0
    for audio_id in all_ids:
        ref_segments = ref_dict.get(audio_id, [])
        pred_segments = pred_dict.get(audio_id, [])
        f1 = compute_frame_f1(ref_segments, pred_segments)
        # Format timestamps to three decimal places.
        ref_timestamps = [f"{s:.3f}-{e:.3f}" for s, e in ref_segments]
        pred_timestamps = [f"{s:.3f}-{e:.3f}" for s, e in pred_segments]
        results.append({
            "id": audio_id,
            "ref_timestamp": ref_timestamps,
            "pred_timestamp": pred_timestamps,
            "seg_f1": round(f1, 4)
        })
        total_f1 += f1

    avg_f1 = total_f1 / len(results) if results else 0.0

    # Write the output JSONL.
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        # Append the summary row.
        f.write(json.dumps({"avg_seg_f1": round(avg_f1, 4)}, ensure_ascii=False) + '\n')

    print(f"Processed {len(results)} audio files; mean F1 = {avg_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute frame-level SpeechF1.")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=float, default=0.04)
    args = parser.parse_args()
    main(args.prediction, args.reference, args.output)
