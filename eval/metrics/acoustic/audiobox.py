import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from audiobox_aesthetics.infer import initialize_predictor


def resolve_model_path(model_path):
    """Resolve a local checkpoint or download a Hugging Face model repository."""
    local_path = Path(model_path).expanduser()
    if local_path.is_file():
        return str(local_path.resolve())
    if local_path.is_dir():
        candidates = sorted(local_path.rglob("checkpoint.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint.pt found in local AudioBox directory: {local_path}"
            )
        return str(candidates[0])
    if local_path.is_absolute():
        raise FileNotFoundError(f"Local AudioBox path does not exist: {local_path}")

    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(str(model_path)))
    candidates = sorted(root.rglob("checkpoint.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint.pt found in Hugging Face repository: {model_path}"
        )
    return str(candidates[0])

# Supported audio extensions (case-insensitive).
AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.opus'}

def collect_audio_files(root_dir):
    """Recursively collect audio files."""
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Directory does not exist: {root}")
    file_paths = []
    for f in root.rglob('*'):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            file_paths.append(str(f))
    return file_paths

def main(input_dir, output_jsonl, model_path, batch_size=32):
    # 1. Collect audio files.
    print(f"Scanning directory: {input_dir}")
    audio_paths = collect_audio_files(input_dir)
    print(f"Found {len(audio_paths)} audio files")

    if not audio_paths:
        print("No audio files found; exiting.")
        return

    # 2. Initialize the model.
    print("Loading the AudioBox Aesthetics model...")
    predictor = initialize_predictor(resolve_model_path(model_path))

    # 3. Prepare the output file.
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Store metrics for the summary.
    all_metrics = {'CE': [], 'CU': [], 'PC': [], 'PQ': []}

    # 4. Run batched inference.
    print("Running inference...")
    with open(out_path, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(audio_paths), batch_size), desc="Batches"):
            batch_paths = audio_paths[i:i+batch_size]
            # Build the list-of-dicts model input.
            inputs = [{"path": p} for p in batch_paths]
            try:
                batch_results = predictor.forward(inputs)  # Returns a list of dicts.
            except Exception as e:
                print(f"Batch failed (indices {i}-{i+batch_size}): {e}")
                continue

            # Process each result.
            for path, result in zip(batch_paths, batch_results):
                # Use the filename stem as the sample ID.
                file_id = Path(path).stem
                # Extract the four metrics.
                ce = result.get('CE')
                cu = result.get('CU')
                pc = result.get('PC')
                pq = result.get('PQ')
                # Skip samples with incomplete metrics.
                if None in (ce, cu, pc, pq):
                    print(f"Warning: incomplete metrics for {path}; skipping")
                    continue

                # Add values to the summary lists.
                all_metrics['CE'].append(ce)
                all_metrics['CU'].append(cu)
                all_metrics['PC'].append(pc)
                all_metrics['PQ'].append(pq)

                # Write the current row.
                line = {
                    "id": file_id,
                    "CE": ce,
                    "CU": cu,
                    "PC": pc,
                    "PQ": pq
                }
                f_out.write(json.dumps(line, ensure_ascii=False) + '\n')

    # 5. Compute summary statistics.
    if all_metrics['CE']:
        summary = {
            "CE_avg": float(np.mean(all_metrics['CE'])),
            "CE_std": float(np.std(all_metrics['CE'], ddof=1)),
            "CU_avg": float(np.mean(all_metrics['CU'])),
            "CU_std": float(np.std(all_metrics['CU'], ddof=1)),
            "PC_avg": float(np.mean(all_metrics['PC'])),
            "PC_std": float(np.std(all_metrics['PC'], ddof=1)),
            "PQ_avg": float(np.mean(all_metrics['PQ'])),
            "PQ_std": float(np.std(all_metrics['PQ'], ddof=1)),
            "num_samples": len(all_metrics['CE'])
        }
        # Append the summary row.
        with open(out_path, 'a', encoding='utf-8') as f_out:
            f_out.write(json.dumps(summary, ensure_ascii=False) + '\n')
        print(f"\nSummary appended for {summary['num_samples']} valid samples")
    else:
        print("No samples were processed successfully; no summary row was written.")

    print(f"Completed; results saved to: {out_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute AudioBox acoustic metrics.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(args.input_dir, args.output, args.model, args.batch_size)
