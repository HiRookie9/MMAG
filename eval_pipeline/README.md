# MMAG Evaluation Pipeline

MMAG is a benchmark for mixed-audio generation. It evaluates generated scenes
containing speech, music, and sound effects, including speaker identity and
temporal alignment control.

The pipeline is split by reported metric rather than by model implementation:

```text
metrics/acoustic/           Distributional similarity and perceptual quality
metrics/speech/             Whisper, WER, speaker similarity, and UTMOSv2
metrics/semantic/           CLAP and AnyAudio-Judge (AAJ)
metrics/temporal/           WhisperX, PE-A-Frame, SpeechF1, and SoundF1
```

## Metrics

| Preset | Metrics |
| --- | --- |
| `overall` | FD, KL, IS, PC, PQ, WER, UTMOSv2, CLAP, AAJ |
| `voice-prompt` | FD, KL, IS, PC, PQ, WER, SPK-SIM, UTMOSv2, CLAP, AAJ |
| `timestamp` | FD, KL, IS, PC, PQ, WER, UTMOSv2, CLAP, AAJ, SpeechF1, SoundF1 |

The custom preset supports both high-level groups and individual pipelines.
The high-level groups expand as follows:

```bash
METRICS=(acoustic speech semantic temporal)
```

For finer control, list individual pipelines instead:

```bash
METRICS=(distribution wer clap speech_f1)
```

Available groups and individual pipelines:

| Selection level | Name | Pipelines |
| --- | --- | --- |
| Group | `acoustic` | `distribution`, `perception` |
| Group | `speech` | `wer`, `speaker-sim`, `utmos` |
| Group | `semantic` | `clap`, `aaj` |
| Group | `temporal` | `sound_f1`, `speech_f1` |
| Individual | `distribution` | FAD, FD, IS, KL |
| Individual | `perception` | AudioBox aesthetics (PC, PQ) |
| Individual | `wer`, `speaker-sim`, `utmos` | Corresponding speech metric |
| Individual | `clap`, `aaj` | Corresponding semantic metric |
| Individual | `sound_f1`, `speech_f1` | Corresponding temporal metric |

Edit the `METRICS` array in `configs/custom.sh`. For example, selecting only
`distribution` does not run AudioBox, while selecting `acoustic` runs both
distribution and perception pipelines. `sound_f1` can use `SOUND_PREDICTION`
as an existing prediction file, or generate predictions with `SOUND_MODEL`.

The individual metric families are:

* Distributional similarity: FAD, FD, KL, and IS.
* Perceptual quality: AudioBox aesthetics (PC and PQ).
* Speech: WER, speaker similarity, and UTMOSv2.
* Semantic: CLAP and AAJ.
* Temporal: SpeechF1 and SoundF1.

FAD is the FAD score computed with VGGish features (`FD_Vggish` in the
implementation), while FD is the Fréchet distance computed with the CNN14
features described in the paper appendix (`FD_CNN14`). The exact labels are
preserved in the output produced by `audioldm_eval`.

## Running the presets

Set input paths once in `configs/paths.sh`, then select a preset:

```bash
# Edit configs/paths.sh and fill in AUDIO_DIR, REF_AUDIO, REF_CAPTION, etc.
bash configs/overall.sh
bash configs/voice_prompt.sh
bash configs/timestamp.sh
bash configs/custom.sh
```

The shared file contains empty values for dataset-specific paths and default
Hugging Face model IDs. Each preset loads it automatically and sets its own
`OUT_DIR`. Edit `configs/custom.sh` to choose metric groups for the custom
preset. Any shared value can also be overridden by an environment variable:

```bash
AUDIO_DIR=/data/generated \
REF_AUDIO=/data/reference_audio.jsonl \
REF_CAPTION=/data/captions.jsonl \
TEXT_REFERENCE=/data/transcripts.jsonl \
SEMANTIC_MANIFEST=/data/semantic.jsonl \
RUBRIC_DIR=/data/rubrics bash configs/overall.sh
```

The default model identifiers are:

* Whisper: `openai/whisper-large-v3`.
* WhisperX: `Systran/faster-whisper-large-v3`.
* PE-A-Frame: `facebook/pe-a-frame-large`.
* AAJ: `cucl2/AnyAudio-Judge-30B`.
* UTMOSv2: leave `UTMOS_CHECKPOINT` empty to use the package's pretrained
  loader.

AudioBox uses the Hugging Face model `facebook/audiobox-aesthetics`; the
pipeline downloads the repository and locates `checkpoint.pt` automatically.
Set `AUDIOBOX_MODEL` to a local checkpoint to override it.

CLAP uses one checkpoint, configured by `CLAP_MODEL_PATH`. Its default is
`lukewys/laion_clap` / `630k-audioset-best.pt`, downloaded through the Hugging
Face cache. Set `CLAP_MODEL_PATH` to a local checkpoint to override it.

All model variables accept either a local path or a model identifier supported
by the underlying loader.

The example above uses the default AAJ model. Set `JUDGE_MODEL` only when you
want to override that default with another Hugging Face ID or a local path.

## Input formats

All manifests are UTF-8 JSONL. Audio IDs must match the audio filename stem;
for example, `abc.wav` corresponds to `{"id": "abc"}`.

* Reference audio: `{"audio_id": "abc", "audio": "/data/ref/abc.wav"}`.
* Captions: `{"audio_id": "abc", "caption": "a dog barks"}`.
* Speech reference: `{"id": "abc", "text": "reference transcript"}`.
* Speaker pairs: `{"prompt_audio": "/data/prompt.wav", "target_audio": "/data/generated.wav"}`.
* Timestamp reference: `{"id": "abc", "timestamp": "1.20-2.80", "caption": "..."}`.
* WhisperX prediction: `{"id": "abc", "segments": [{"start": 1.2, "end": 2.8}]}`.
* SoundF1 prediction: `id`, `pred_timestamp`, and `original_timestamp` fields.
* AAJ manifest: at least an `id` field and a matching `{id}.jsonl` rubric file.

## Project information

MMAG evaluates mixed-audio generation under multiple control conditions,
including acoustic scenes, speaker identity, and temporal alignment.

Evaluated systems include [AudioDirector](https://github.com/Riddae/AuDirector),
[LTX-2.3](https://github.com/Lightricks/LTX-2),
[Ovi 1.1](https://github.com/character-ai/Ovi),
[MOVA](https://github.com/OpenMOSS/MOVA),
[JavisDiT++](https://github.com/JavisVerse/JavisDiT),
[UniAVGen](https://github.com/MCG-NJU/Sora2-mini),
[Dasheng-AudioGen](https://github.com/xiaomi-research/dasheng-audiogen), and
Ming-Omni-TTS.

This project acknowledges the authors of
[MECAT](https://github.com/xiaomi-research/mecat). See the repository-level
license for redistribution terms.
