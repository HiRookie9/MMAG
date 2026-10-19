import os
import json
import argparse
import wespeaker
import torch
from tqdm import tqdm


def move_model(model, device):
    """Set WeSpeaker's device and synchronize its underlying torch module."""
    device = str(device)
    model.set_device(device)
    torch_model = getattr(model, "model", None)
    if isinstance(torch_model, torch.nn.Module):
        torch_model.to(torch.device(device))


def main(input_jsonl, output_jsonl, model_name="english", device=None):
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU")
        device = "cpu"
    print(f"Using device: {device}")

    model = wespeaker.load_model(model_name)
    move_model(model, device)

    with open(input_jsonl, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    results = []
    similarities = []

    with open(input_jsonl, "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(tqdm(fin, total=total_lines, desc="Processing"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                prompt_audio = data.get("prompt_audio", data.get("reference_audio"))
                target_audio = data.get("target_audio", data.get("audio"))
                # score = data.get("score", "")

                if not prompt_audio or not target_audio:
                    print(f"Warning: line {line_num} missing audio paths, skip")
                    continue
                if not os.path.isfile(prompt_audio):
                    print(f"Warning: prompt audio not found: {prompt_audio}, skip")
                    continue
                if not os.path.isfile(target_audio):
                    print(f"Warning: target audio not found: {target_audio}, skip")
                    continue

                try:
                    similarity = model.compute_similarity(prompt_audio, target_audio)
                except RuntimeError as error:
                    message = str(error)
                    is_device_mismatch = (
                        "Input type" in message and "weight type" in message
                    ) or ("Expected all tensors to be on the same device" in message)
                    if not is_device_mismatch or str(device) == "cpu":
                        raise
                    print(
                        f"Device mismatch on line {line_num}; moving WeSpeaker to CPU and retrying"
                    )
                    device = "cpu"
                    move_model(model, device)
                    similarity = model.compute_similarity(prompt_audio, target_audio)
                similarities.append(similarity)

                base_name = os.path.basename(prompt_audio)
                audio_id = os.path.splitext(base_name)[0]

                results.append({
                    "id": audio_id,
                    "sim": round(similarity, 6),
                    # "score": score
                })
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    # Compute and print the mean similarity.
    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        print(f"Mean speaker similarity ({len(similarities)} pairs): {avg_sim:.6f}")
    else:
        print("No speaker similarities were computed.")

    # Write per-pair output.
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        # last line: avg_sim summary
        fout.write(json.dumps({
            # "id": "AVG",
            "avg_sim": round(avg_sim, 6) if avg_sim is not None else None,
            "num_pairs": len(similarities),
        }, ensure_ascii=False) + "\n")


    print(f"Done. Processed {len(results)} pairs. Output saved to {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute speaker similarity with WeSpeaker.")
    parser.add_argument("--pairs", required=True, help="JSONL with prompt_audio/target_audio")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="english")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    main(args.pairs, args.output, args.model, args.device)
