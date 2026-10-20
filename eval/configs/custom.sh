#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/configs/paths.sh"

# Groups: acoustic speech semantic temporal.
# Individual metrics: distribution perception wer speaker-sim utmos clap aaj sound_f1 speech_f1.
METRICS=(acoustic speech semantic temporal)
OUT_DIR="${OUT_DIR:-/tmp/audio_eval/custom}"

exec "$ROOT/run_eval.sh" custom --metrics "${METRICS[@]}" \
  --audio-dir "$AUDIO_DIR" --output-dir "$OUT_DIR" \
  --ref-audio "$REF_AUDIO" --ref-caption "$REF_CAPTION" \
  --text-reference "$TEXT_REFERENCE" --whisper-model "$WHISPER_MODEL" \
  --speaker-pairs "$SPEAKER_PAIRS" --timestamp-reference "$TIMESTAMP_REFERENCE" \
  --sound-prediction "$SOUND_PREDICTION" --whisperx-model "$WHISPERX_MODEL" \
  --sound-model "$SOUND_MODEL" \
  --audiobox-model "$AUDIOBOX_MODEL" \
  --semantic-manifest "$SEMANTIC_MANIFEST" --rubric-dir "$RUBRIC_DIR" \
  --judge-model "$JUDGE_MODEL" --utmos-checkpoint "$UTMOS_CHECKPOINT" \
  --clap-model "$CLAP_MODEL_PATH" --device "$DEVICE"
