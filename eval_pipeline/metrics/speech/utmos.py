import os
# These thread limits must be set before importing torch.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import utmosv2
import json
import argparse
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser(description="UTMOSv2 MOS prediction with optional filtering")
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("output_jsonl", help="Output JSONL file path")
    parser.add_argument("--input_jsonl", default=None, help="Optional JSONL file with 'id' field to filter audio files")
    parser.add_argument("--checkpoint", default=None, help="Optional UTMOSv2 checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load the pretrained model, optionally from a local checkpoint.
    model = utmosv2.create_model(pretrained=True, checkpoint_path=args.checkpoint)
    model = model.to(args.device)

    # Collect common audio formats recursively.
    audio_extensions = {'.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg'}
    all_files = []
    for ext in audio_extensions:
        all_files.extend(Path(args.input_dir).rglob(f"*{ext}"))
    all_files = [str(p) for p in all_files]

    if args.input_jsonl:
        ids = set()
        with open(args.input_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if 'id' in item:
                        ids.add(str(item['id']))
        all_files = [f for f in all_files if Path(f).stem in ids]
    print(f"Processing {len(all_files)} audio files")

    # if not all_files:
    #     print("No matching audio files; exiting.")
    #     return

    # Predict MOS values.
    batch_size = 1
    num_workers = 8
    mos_list = model.predict(input_dir=args.input_dir, batch_size=batch_size, num_workers=num_workers)

    # Compute the mean MOS.
    avg_mos = sum(item['predicted_mos'] for item in mos_list) / len(mos_list) if mos_list else 0.0

    # Save per-file results and the summary row.
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in mos_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary = {"file_path": args.input_dir, "avg_mos": avg_mos}
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Results saved to {os.path.abspath(args.output_jsonl)}")

if __name__ == "__main__":
    main()
