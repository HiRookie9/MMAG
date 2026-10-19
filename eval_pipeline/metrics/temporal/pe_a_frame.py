import json
import os
import argparse
from collections import defaultdict
from transformers import PeAudioFrameLevelModel, PeAudioProcessor
import torch
from tqdm import tqdm

def extract_true_segments(preds, time_resolution=0.04):
    """
    Convert an (N, L) boolean tensor into N lists of time intervals.
    """
    if preds.dim() == 3 and preds.size(0) == 1:
        preds = preds.squeeze(0)
    elif preds.dim() > 2:
        raise ValueError(f"Unexpected preds shape: {preds.shape}")

    N, L = preds.shape
    results = []
    for i in range(N):
        segments = []
        in_seg = False
        start_idx = 0
        for j in range(L):
            if preds[i, j] and not in_seg:
                in_seg = True
                start_idx = j
            elif not preds[i, j] and in_seg:
                in_seg = False
                end_idx = j - 1
                segments.append((
                    round(start_idx * time_resolution, 2),
                    round((end_idx + 1) * time_resolution, 2)
                ))
        if in_seg:
            segments.append((
                round(start_idx * time_resolution, 2),
                round(L * time_resolution, 2)
            ))
        results.append(segments)
    return results


def main(audio_folder, jsonl_path, output_jsonl_path, model_path, threshold=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model and processor once.
    model = PeAudioFrameLevelModel.from_pretrained(model_path)
    processor = PeAudioProcessor.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # Read the JSONL manifest.
    entries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # Group entries by audio ID; one audio may have multiple captions.
    groups = defaultdict(list)
    for entry in entries:
        groups[entry['id']].append(entry)

    output_entries = []

    for audio_id, group in tqdm(groups.items(), desc="Processing audios"):
        audio_path_wav = os.path.join(audio_folder, f"{audio_id}.wav")
        audio_path_flac = os.path.join(audio_folder, f"{audio_id}.flac")

        if os.path.exists(audio_path_wav):
            audio_path = audio_path_wav
        elif os.path.exists(audio_path_flac):
            audio_path = audio_path_flac
        else:
            print(f"Warning: audio file {audio_id} (wav/flac) not found, skipping.")
            continue

        captions = [item['caption'] for item in group]
        print(f"Processing {audio_id} with {len(captions)} captions.")

        # Extract audio features.
        audio_inputs = processor.feature_extractor(
            audio_path, sampling_rate=48_000, return_tensors="pt"
        ).to(device)
        # Tokenize captions.
        text_inputs = processor.tokenizer(
            captions, padding=True, return_tensors="pt"
        ).to(device)
        inputs = {**audio_inputs, **text_inputs}

        with torch.inference_mode():
            outputs = model(**inputs, return_spans=True)

        logits_audio_text = outputs["logits_audio_text"]  # shape: (num_captions, num_frames)
        probs = torch.sigmoid(logits_audio_text)
        preds = probs > threshold

        # Extract predicted intervals for each caption.
        segments_list = extract_true_segments(preds, time_resolution=0.04)

        # Assemble output rows.
        for entry, segs in zip(group, segments_list):
            new_entry = {
                "id": entry['id'],
                "caption": entry['caption'],
                "pred_timestamp": [[s, e] for s, e in segs]
            }
            # Preserve the reference timestamp when available.
            if 'timestamp' in entry:
                new_entry['original_timestamp'] = entry['timestamp']
            output_entries.append(new_entry)

    # Write the output JSONL.
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in output_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Done. Output saved to {output_jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sound event timestamps with PE-A-Frame.")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    main(args.audio_dir, args.manifest, args.output, args.model, args.threshold)
