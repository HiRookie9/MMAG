import json
import os
import sys
import argparse
from anyaudio_judge import AnyAudioJudge
from tqdm import tqdm
import torch
from accelerate import dispatch_model, infer_auto_device_map
# num_layers = 48
# layers_per_gpu = 6  # 48 / 8

# device_map = {
#     "thinker.audio_tower": 0,
#     "thinker.visual": 0,
#     "thinker.model.embed_tokens": 0,
# }
# for i in range(num_layers):
#     gpu = i // layers_per_gpu
#     device_map[f"thinker.model.layers.{i}"] = gpu
# device_map.update({
#     "thinker.model.norm": 7,
#     "thinker.model.rotary_emb": 7,
#     "thinker.lm_head": 7,
# })

# layer_counts = [3, 7, 7, 7, 7, 7, 6, 4]
# assert sum(layer_counts) == 48

# device_map = {
#     "thinker.audio_tower": 0,
#     "thinker.visual": 0,
#     "thinker.model.embed_tokens": 0,
# }

# layer_idx = 0
# for gpu, count in enumerate(layer_counts):
#     for _ in range(count):
#         device_map[f"thinker.model.layers.{layer_idx}"] = gpu
#         layer_idx += 1

# device_map.update({
#     "thinker.model.norm": 7,
#     "thinker.model.rotary_emb": 7,
#     "thinker.lm_head": 7,
# })

# Optional custom device-map loading can be added here if needed.
# judge = AnyAudioJudge.from_pretrained(
#     MODEL_PATH,
#     device_map=device_map,
#     #torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
# )
def load_rubric(rubric_path):
    rubric = []
    with open(rubric_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rubric.append(json.loads(line))
    return rubric

def find_audio_file(base_dir, sample_id):
    candidates = [f"{sample_id}.wav", f"{sample_id}.flac"]
    for cand in candidates:
        full_path = os.path.join(base_dir, cand)
        if os.path.exists(full_path):
            return full_path
    return None

def process_sample(sample, audio_dir, rubric_dir, judge):
    sample_id = sample['id']
    audio_path = find_audio_file(audio_dir, sample_id)
    rubric_path = os.path.join(rubric_dir, f"{sample_id}.jsonl")
    
    if audio_path is None:
        print(f"Warning: audio file not found for id={sample_id} (tried .wav and .flac)")
        return None
    if not os.path.exists(rubric_path):
        print(f"Warning: rubric file does not exist: {rubric_path}")
        return None
    
    rubric = load_rubric(rubric_path)
    if not rubric:
        print(f"Warning: rubric file is empty: {rubric_path}")
        return None
    
    result = judge.judge(audio_path, rubric, mode="logits")
    
    details = []
    for item, rub_item in zip(result.items, rubric):
        details.append({
            "dimension": rub_item.get("dimension", ""),
            "question": rub_item.get("question", ""),
            "p_yes": item.p_yes
        })
    
    return {
        "id": sample_id,
        "alignment_score": result.score,
        "details": details
    }

def main():
    parser = argparse.ArgumentParser(description="Batch audio evaluation with one JSON result per sample.")
    parser.add_argument("--audio-dir", required=True, help="Directory containing generated audio")
    parser.add_argument("--manifest", required=True, help="JSONL manifest with id fields")
    parser.add_argument("--rubric-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    # Create the output directory before loading the judge.
    os.makedirs(args.output_dir, exist_ok=True)
    judge = AnyAudioJudge.from_pretrained(args.model, dtype="auto", device_map="auto")
    
    # Read the sample manifest.
    samples = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples")
    
    success_count = 0
    for sample in tqdm(samples, desc="Evaluating samples", unit="sample"):
        res = process_sample(sample, args.audio_dir, args.rubric_dir, judge)
        if res is None:
            continue
        
        # Store one JSON result per sample.
        out_path = os.path.join(args.output_dir, f"{res['id']}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        success_count += 1
    
    print(f"Completed {success_count} samples; results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
