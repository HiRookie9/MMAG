#!/usr/bin/env bash
# Shared input paths and model settings for the MMAG evaluation presets.
# Edit the empty data paths below, or override them with environment variables.

AUDIO_DIR="${AUDIO_DIR:-}"
REF_AUDIO="${REF_AUDIO:-}"
REF_CAPTION="${REF_CAPTION:-}"
TEXT_REFERENCE="${TEXT_REFERENCE:-}"
SPEAKER_PAIRS="${SPEAKER_PAIRS:-}"
TIMESTAMP_REFERENCE="${TIMESTAMP_REFERENCE:-}"
SOUND_PREDICTION="${SOUND_PREDICTION:-}"
SEMANTIC_MANIFEST="${SEMANTIC_MANIFEST:-}"
RUBRIC_DIR="${RUBRIC_DIR:-}"

# Hugging Face IDs are used by default; replace with local model paths if needed.
WHISPER_MODEL="${WHISPER_MODEL:-openai/whisper-large-v3}"
WHISPERX_MODEL="${WHISPERX_MODEL:-Systran/faster-whisper-large-v3}"
SOUND_MODEL="${SOUND_MODEL:-facebook/pe-a-frame-large}"
JUDGE_MODEL="${JUDGE_MODEL:-cucl2/AnyAudio-Judge-30B}"
AUDIOBOX_MODEL="${AUDIOBOX_MODEL:-facebook/audiobox-aesthetics}"

# Leave empty to use the package's pretrained UTMOSv2 loader.
UTMOS_CHECKPOINT="${UTMOS_CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda}"

# Hugging Face checkpoint URLs are downloaded into the local cache by the CLAP metric.
export CLAP_MODEL_PATH="${CLAP_MODEL_PATH:-https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt}"
