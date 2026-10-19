#!/usr/bin/env python3
"""Compute text-audio CLAP similarity for generated audio."""
import argparse
import json
import os
from urllib.parse import unquote, urlparse
from pathlib import Path

import librosa
import laion_clap
import numpy as np
import torch


def resolve_checkpoint(value):
    """Return a local checkpoint path, downloading Hugging Face URLs on demand."""
    if not value.startswith(("http://", "https://")):
        return value
    parsed = urlparse(value)
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if "huggingface.co" in parsed.netloc and "resolve" in parts:
        index = parts.index("resolve")
        repo_id = "/".join(parts[:index])
        filename = "/".join(parts[index + 2:])
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename, revision=parts[index + 1])
    raise ValueError(f"Unsupported checkpoint URL: {value}")


def read_captions(path):
    captions = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                key = item.get("audio_id", item.get("id"))
                text = item.get("caption", item.get("text"))
                if key is not None and text:
                    captions[str(key)] = text
    return captions


def audio_map(directory):
    extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
    return {path.stem: str(path) for path in Path(directory).rglob("*")
            if path.is_file() and path.suffix.lower() in extensions}


def score(model, audio, text):
    with torch.no_grad():
        text_embedding = model.get_text_embedding([text], use_tensor=False)
        audio_embedding = model.get_audio_embedding_from_data(x=[audio], use_tensor=False)
    numerator = float(np.sum(audio_embedding[0] * text_embedding[0]))
    denominator = float(np.linalg.norm(audio_embedding[0]) * np.linalg.norm(text_embedding[0]))
    return numerator / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser(description="Compute CLAP semantic similarity.")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("CLAP_MODEL_PATH"), required=False)
    args = parser.parse_args()
    if not args.model:
        raise SystemExit("Set --model or CLAP_MODEL_PATH to a LAION-CLAP checkpoint")

    captions = read_captions(args.captions)
    audios = audio_map(args.audio_dir)
    checkpoint = resolve_checkpoint(args.model)
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt(ckpt=checkpoint, verbose=False)
    model.eval()

    rows = []
    for sample_id in sorted(captions.keys() & audios.keys()):
        audio, _ = librosa.load(audios[sample_id], sr=48000)
        rows.append({"id": sample_id, "CLAP_score": score(model, audio, captions[sample_id])})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    means = {
        "CLAP_score": float(np.mean([r["CLAP_score"] for r in rows])) if rows else 0.0,
        "num_samples": len(rows),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.write(json.dumps(means, ensure_ascii=False) + "\n")
    print(json.dumps(means, ensure_ascii=False))


if __name__ == "__main__":
    main()
