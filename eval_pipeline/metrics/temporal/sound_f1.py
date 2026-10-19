import json
import sys
import argparse
from collections import defaultdict

def parse_timestamp(ts_str):
    s, e = ts_str.split('-')
    return float(s), float(e)

def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_int = sorted(intervals, key=lambda x: x[0])
    merged = []
    cur_s, cur_e = sorted_int[0]
    for s, e in sorted_int[1:]:
        if s <= cur_e:          # Overlap or adjacency.
            if e > cur_e:
                cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

def clip_interval(interval, low=0, high=10):
    s, e = interval
    s = max(s, low)
    e = min(e, high)
    if s < e:
        return (s, e)
    return None

def compute_frame_counts(ref_segments, pred_segments, resolution=0.04):
    """
    Return (tp, fp, fn, f1).
    The group F1 is reported for reference; global statistics use TP/FP/FN.
    """
    # Merge intervals defensively.
    ref_segments = merge_intervals(ref_segments)
    pred_segments = merge_intervals(pred_segments)

    max_time = 0.0
    for s, e in ref_segments + pred_segments:
        if e > max_time:
            max_time = e
    if max_time == 0:
        # Empty reference and prediction are treated as a perfect match.
        return 0, 0, 0, 1.0

    n_frames = int(max_time / resolution) + 1
    ref_binary = [0] * n_frames
    pred_binary = [0] * n_frames
    half_res = resolution * 0.5

    for i in range(n_frames):
        frame_start = i * resolution
        frame_end = frame_start + resolution

        ref_overlap = 0.0
        for s, e in ref_segments:
            ov = max(0.0, min(frame_end, e) - max(frame_start, s))
            ref_overlap += ov
        if ref_overlap >= half_res:
            ref_binary[i] = 1

        pred_overlap = 0.0
        for s, e in pred_segments:
            ov = max(0.0, min(frame_end, e) - max(frame_start, s))
            pred_overlap += ov
        if pred_overlap >= half_res:
            pred_binary[i] = 1

    tp = sum(1 for r, p in zip(ref_binary, pred_binary) if r and p)
    fp = sum(1 for r, p in zip(ref_binary, pred_binary) if not r and p)
    fn = sum(1 for r, p in zip(ref_binary, pred_binary) if r and not p)

    if tp == 0:
        return tp, fp, fn, 0.0

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return tp, fp, fn, 0.0
    f1 = 2 * prec * rec / (prec + rec)
    return tp, fp, fn, f1

def group_by_caption_similarity(rows):
    """Group rows by matching or near-matching captions."""
    n = len(rows)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i+1, n):
            cap_i = rows[i]['caption'].lower()
            cap_j = rows[j]['caption'].lower()
            if cap_i == cap_j or cap_i in cap_j or cap_j in cap_i:
                union(i, j)

    groups = defaultdict(list)
    for i, row in enumerate(rows):
        root = find(i)
        groups[root].append(row)
    return list(groups.values())

def main(input_file, output_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    # Group by audio ID.
    id_rows = defaultdict(list)
    for item in data:
        id_rows[item['id']].append(item)

    results = []

    # Global accumulators for micro-averaged F1.
    global_tp = 0
    global_fp = 0
    global_fn = 0

    for audio_id, rows in id_rows.items():
        groups = group_by_caption_similarity(rows)
        for group in groups:
            # Merge all predicted intervals.
            all_pred = []
            for row in group:
                all_pred.extend([tuple(p) for p in row['pred_timestamp']])

            # Merge all reference intervals.
            all_ref = []
            for row in group:
                s, e = parse_timestamp(row['original_timestamp'])
                all_ref.append((s, e))

            # Clip intervals to the ten-second evaluation window.
            clipped_pred = []
            for s, e in all_pred:
                clipped = clip_interval((s, e), 0, 10)
                if clipped:
                    clipped_pred.append(clipped)
            clipped_ref = []
            for s, e in all_ref:
                clipped = clip_interval((s, e), 0, 10)
                if clipped:
                    clipped_ref.append(clipped)

            merged_pred = merge_intervals(clipped_pred)
            merged_ref = merge_intervals(clipped_ref)

            # Compute group TP, FP, FN, and F1 for reporting.
            tp, fp, fn, f1 = compute_frame_counts(merged_ref, merged_pred)

            # Add counts to the global totals.
            global_tp += tp
            global_fp += fp
            global_fn += fn

            # Keep the group-level F1 in the output row.
            captions = sorted(set(row['caption'] for row in group))
            group_label = '|'.join(captions)

            results.append({
                'id': audio_id,
                'caption_group': group_label,
                'ref_timestamp': [f"{s:.3f}-{e:.3f}" for s, e in merged_ref],
                'pred_timestamp': [f"{s:.3f}-{e:.3f}" for s, e in merged_pred],
                'seg_f1': round(f1, 4)
            })

    # Compute micro-averaged F1 from global counts.
    if global_tp == 0:
        global_f1 = 0.0
    else:
        global_prec = global_tp / (global_tp + global_fp)
        global_rec = global_tp / (global_tp + global_fn)
        global_f1 = 2 * global_prec * global_rec / (global_prec + global_rec) if (global_prec + global_rec) > 0 else 0.0

    # Write the output JSONL.
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        # Append the global micro-averaged F1 as the final row.
        f.write(json.dumps({"avg_seg_f1": round(global_f1, 4)}, ensure_ascii=False) + '\n')

    print(f"Processed {len(results)} groups; global micro F1 = {global_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute frame-level SoundF1 from predicted segments.")
    parser.add_argument("--input", required=True, help="JSONL with id, pred_timestamp and original_timestamp")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)
