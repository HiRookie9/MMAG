import os

# PyTorch 2.6+ compatibility for trusted WhisperX/Pyannote checkpoints.
# This must be set before importing torch or whisperx.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import gc
import inspect
import json
from pathlib import Path

import torch
import whisperx
from tqdm import tqdm
from whisperx.alignment import load_align_model


BATCH_SIZE = 16
COMPUTE_TYPE = "float16"
LANGUAGE = "en"


def load_alignment_model(device, local_files_only):
    """
    Load the alignment model while supporting both old and new WhisperX
    versions.

    Older WhisperX versions do not support model_cache_only.
    Newer versions support model_cache_only rather than local_files_only.
    """
    kwargs = {}

    signature = inspect.signature(load_align_model)
    if "model_cache_only" in signature.parameters:
        kwargs["model_cache_only"] = local_files_only

    return load_align_model(
        LANGUAGE,
        device,
        **kwargs,
    )


def load_target_ids(jsonl_path):
    target_ids = set()

    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                print(
                    f"Warning: skipping invalid JSON on line "
                    f"{line_number}: {error}"
                )
                continue

            if "id" in data and data["type"]=="speech":
                # Audio filename stems are strings.
                target_ids.add(str(data["id"]))

    return target_ids


def clean_word(word):
    """
    Convert WhisperX word data into JSON-serializable values.

    Some words may not contain timestamps when forced alignment fails, so
    optional fields are only included when they are available.
    """
    cleaned = {
        "word": word.get("word", ""),
    }

    for key in ("start", "end", "score"):
        value = word.get(key)
        if value is not None:
            cleaned[key] = float(value)

    return cleaned


def process_audio(
    audio_file,
    model,
    align_model,
    metadata,
    device,
):
    audio = whisperx.load_audio(audio_file)

    transcription = model.transcribe(
        audio,
        batch_size=BATCH_SIZE,
    )

    aligned_result = whisperx.align(
        transcription["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    cleaned_segments = []

    for segment in aligned_result["segments"]:
        cleaned_segment = {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": segment["text"],
            "words": [
                clean_word(word)
                for word in segment.get("words", [])
            ],
        }
        cleaned_segments.append(cleaned_segment)

    return cleaned_segments


def main(
    audio_folder,
    jsonl_input,
    jsonl_output,
    model_dir,
    device,
    compute_type,
    local_files_only,
):
    target_ids = load_target_ids(jsonl_input)
    print(f"Loaded {len(target_ids)} target IDs")

    print("Loading WhisperX model...")
    model = whisperx.load_model(
        model_dir,
        device,
        compute_type=compute_type,
        language=LANGUAGE,
        local_files_only=local_files_only,
    )

    print("Loading alignment model...")
    align_model, metadata = load_alignment_model(
        device=device,
        local_files_only=local_files_only,
    )

    audio_directory = Path(audio_folder)

    audio_files = sorted(
        list(audio_directory.glob("*.wav"))
        + list(audio_directory.glob("*.flac"))
    )
    print(f"Found {len(audio_files)} audio files")

    files_to_process = [
        audio_file
        for audio_file in audio_files
        if audio_file.stem in target_ids
    ]
    print(f"Matched {len(files_to_process)} target IDs")

    output_path = Path(jsonl_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    failed_count = 0

    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            progress = tqdm(
                files_to_process,
                desc="Processing audio",
                unit="file",
            )

            for audio_path in progress:
                file_id = audio_path.stem

                try:
                    segments = process_audio(
                        audio_file=str(audio_path),
                        model=model,
                        align_model=align_model,
                        metadata=metadata,
                        device=device,
                    )

                    result_line = {
                        "id": file_id,
                        "segments": segments,
                    }

                    output_file.write(
                        json.dumps(result_line, ensure_ascii=False) + "\n"
                    )
                    output_file.flush()

                    processed_count += 1
                    tqdm.write(
                        f"Completed: {file_id} "
                        f"({processed_count} successful)"
                    )

                except Exception as error:
                    failed_count += 1
                    tqdm.write(
                        f"Error: {file_id}: "
                        f"{type(error).__name__}: {error}"
                    )

    finally:
        del model
        del align_model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"Completed {processed_count} files; "
        f"{failed_count} files failed; "
        f"results saved to {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run WhisperX speech timestamp extraction."
    )

    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Directory containing WAV or FLAC files.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Input JSONL manifest containing target IDs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Whisper/Faster-Whisper model name or local directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--compute-type",
        default=COMPUTE_TYPE,
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only use locally cached model files when supported.",
    )

    args = parser.parse_args()

    main(
        audio_folder=args.audio_dir,
        jsonl_input=args.manifest,
        jsonl_output=args.output,
        model_dir=args.model,
        device=args.device,
        compute_type=args.compute_type,
        local_files_only=args.local_files_only,
    )