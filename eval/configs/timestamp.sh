#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/configs/paths.sh"

OUT_DIR="${OUT_DIR:-/tmp/audio_eval/timestamp}"

: "${AUDIO_DIR:?Set AUDIO_DIR to the generated-audio directory}"
: "${REF_AUDIO:?Set REF_AUDIO to the reference-audio JSONL}"
: "${REF_CAPTION:?Set REF_CAPTION to the caption JSONL}"
: "${TEXT_REFERENCE:?Set TEXT_REFERENCE to the transcript JSONL}"
: "${TIMESTAMP_REFERENCE:?Set TIMESTAMP_REFERENCE to the timestamp JSONL}"
: "${SEMANTIC_MANIFEST:?Set SEMANTIC_MANIFEST to the AAJ manifest JSONL}"
: "${RUBRIC_DIR:?Set RUBRIC_DIR to the AAJ rubric directory}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL to an AnyAudioJudge checkpoint or HF ID}"

ARGS=(overall --audio-dir "$AUDIO_DIR" --output-dir "$OUT_DIR" \
  --ref-audio "$REF_AUDIO" --ref-caption "$REF_CAPTION" \
  --text-reference "$TEXT_REFERENCE" --whisper-model "$WHISPER_MODEL" \
  --semantic-manifest "$SEMANTIC_MANIFEST" --rubric-dir "$RUBRIC_DIR" \
  --judge-model "$JUDGE_MODEL" --utmos-checkpoint "$UTMOS_CHECKPOINT" \
  --audiobox-model "$AUDIOBOX_MODEL" --clap-model "$CLAP_MODEL_PATH" --device "$DEVICE")

# Timestamp mode extends overall with SpeechF1 and SoundF1.
ARGS[0]=timestamp
ARGS+=(--timestamp-reference "$TIMESTAMP_REFERENCE" --whisperx-model "$WHISPERX_MODEL" \
  --sound-model "$SOUND_MODEL")
exec "$ROOT/run_eval.sh" "${ARGS[@]}"
