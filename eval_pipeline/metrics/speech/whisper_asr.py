import json
import os
import argparse
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm

def load_whisper_model(model_id: str, device: str):
    """Load Whisper and return an ASR pipeline."""
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=device,
        chunk_length_s=30,
        ignore_warning=True,  # Enable long-audio chunking.
    )
    return asr_pipe

def transcribe_audio(audio_path: str, asr_pipe):
    """
    Transcribe one audio file and downmix multi-channel input to mono.
    """
    waveform, sr = torchaudio.load(audio_path)
    # Downmix multi-channel audio to mono.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio_array = waveform.squeeze().numpy()
    # Request text only; timestamp alignment is handled separately.
    result = asr_pipe({"array": audio_array, "sampling_rate": sr}, return_timestamps=False)
    return result["text"]

def process_jsonl(folder_path: str, jsonl_path: str, output_path: str, model_id: str, device: str):
    """
    Process a manifest of audio IDs with a Whisper model.
    """
    # Load the model once.
    print("Loading Whisper model...")
    asr_pipe = load_whisper_model(model_id, device)
    print("Model loaded.")

    # Read the input JSONL manifest.
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []
    # Show progress while processing the manifest.
    for line in tqdm(lines, desc="Processing audio"):
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError:
            print(f"Warning: invalid JSON line: {line}")
            continue

        # Accept both legacy result=true manifests and the unified format.
        if data.get('result') is False:
            continue

        audio_id = data['id']
        # Try common lossless extensions.
        wav_path = os.path.join(folder_path, f"{audio_id}.wav")
        flac_path = os.path.join(folder_path, f"{audio_id}.flac")
        if os.path.exists(wav_path):
            audio_path = wav_path
        elif os.path.exists(flac_path):
            audio_path = flac_path
        else:
            print(f"Audio file not found for id: {audio_id}")
            continue

        # Run transcription.
        try:
            transcription = transcribe_audio(audio_path, asr_pipe)
            results.append({"id": audio_id, "transcription": transcription})
        except Exception as e:
            print(f"Error transcribing {audio_id}: {e}")
            # Continue with the next sample.

    # Write the output JSONL.
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Done. Processed {len(results)} files. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe generated audio listed in a JSONL manifest.")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--manifest", required=True, help="JSONL with id and optional result=true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, help="Whisper model path or Hugging Face ID")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    process_jsonl(args.audio_dir, args.manifest, args.output, args.model, args.device)
