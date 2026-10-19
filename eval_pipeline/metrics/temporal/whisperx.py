import os
import json
import argparse
import whisperx
import gc
import torch
from pathlib import Path
from tqdm import tqdm

BATCH_SIZE = 16
COMPUTE_TYPE = "float16"

def load_target_ids(jsonl_path):
    ids = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'id' in data:
                    ids.add(data['id'])
            except json.JSONDecodeError:
                continue
    return ids

def process_audio(audio_file, model, align_model, metadata, device):
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=BATCH_SIZE)
    result_aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )
    segments = []
    for seg in result_aligned["segments"]:
        seg_clean = {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"],
            "words": [
                {
                    "word": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "score": float(w["score"])
                }
                for w in seg.get("words", [])
            ]
        }
        segments.append(seg_clean)
    return segments

def main(audio_folder, jsonl_input, jsonl_output, model_dir, device, compute_type, local_files_only):
    # Load target IDs.
    target_ids = load_target_ids(jsonl_input)
    print(f"Loaded {len(target_ids)} target IDs")

    # Load the models once.
    print("Loading WhisperX model...")
    model = whisperx.load_model(
        model_dir,
        device,
        compute_type=compute_type,
        local_files_only=local_files_only
    )
    print("Loading alignment model...")
    align_model, metadata = whisperx.load_align_model(
        language_code='en',
        device=device,
        model_cache_only=local_files_only,
    )

    # Scan the audio directory.
    wav_files = list(Path(audio_folder).glob("*.wav")) + list(Path(audio_folder).glob("*.flac"))
    print(f"Found {len(wav_files)} audio files")

    # Keep only files listed in the manifest.
    to_process = [f for f in wav_files if f.stem in target_ids]
    print(f"Matched {len(to_process)} target IDs")

    # Write the output manifest.
    processed_count = 0
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_output)), exist_ok=True)
    with open(jsonl_output, 'w', encoding='utf-8') as out_f:
        for wav_path in tqdm(to_process, desc="Processing audio", unit="file"):
            file_id = wav_path.stem
            try:
                segments = process_audio(str(wav_path), model, align_model, metadata, device)
                result_line = {"id": file_id, "segments": segments}
                out_f.write(json.dumps(result_line, ensure_ascii=False) + '\n')
                out_f.flush()
                processed_count += 1
                tqdm.write(f"Completed: {file_id} ({processed_count} successful)")
            except Exception as e:
                tqdm.write(f"Error: {file_id}: {e}")
                continue

    # Release model memory.
    del model, align_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Completed {processed_count} files; results saved to {jsonl_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WhisperX speech timestamp extraction.")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-type", default=COMPUTE_TYPE)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    main(args.audio_dir, args.manifest, args.output, args.model, args.device, args.compute_type, args.local_files_only)
