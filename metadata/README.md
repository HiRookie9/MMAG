# MMAG Evaluation Metadata

This directory contains the JSONL metadata used by the MMAG evaluation
pipeline. Set the corresponding paths in`eval/configs/paths.sh in the
MMAG repository. These files describe reference samples; they are separate
from the generated audio in `AUDIO_DIR`.

## Layout

```text
metadata/
├── acoustic/
│   ├── main_set/{audio,caption}.jsonl
│   ├── timestamp_set/{audio,caption}.jsonl
│   └── voice_cloning_set/{audio,caption}.jsonl
├── semantic/aaj_rubric.zip
├── speech/{transcript,spk_sim}.jsonl
└── temporal/timestamp.jsonl
```

Every file is UTF-8 JSONL, with one JSON object per line. `id` or `audio_id`
must match the generated audio filename stem, such as `sample_001` for
`sample_001.wav`.

## Acoustic files

Each acoustic split has an `audio.jsonl` and a matching `caption.jsonl`.

* `audio.jsonl` maps `audio_id` to the reference audio path using `audio`:
  `{"audio_id":"sample_001","audio":"ref_audio/main_set/sample_001.wav"}`.
  Set `REF_AUDIO` to the split used by the evaluation.
* `caption.jsonl` maps `audio_id` to the natural-language description in
  `caption`. Set `REF_CAPTION` to the matching file; it is used by CLAP and
  can also be used by distributional evaluation.

The supplied acoustic audio paths are relative paths. They must resolve in
the evaluation environment; custom manifests may use absolute paths.

## Speech files

`speech/transcript.jsonl` supplies WER references. Required fields are `id`
and `text`; `caption`, `location`, and `idx` are optional metadata. Set
`TEXT_REFERENCE` to this file.

`speech/spk_sim.jsonl` supplies prompt/target pairs for speaker similarity.
The evaluator accepts `prompt_audio` (or `reference_audio`) and
`target_audio` (or `audio`). The checked-in file currently uses the legacy
key `prompt_audio_audio`; rename it to `prompt_audio` before running the
pipeline. Both paths must resolve, or be made absolute. Set `SPEAKER_PAIRS`
to the resulting file.

## Semantic files

`semantic/aaj_rubric.zip` contains AnyAudio-Judge (AAJ) rubric files. Extract
it and set `RUBRIC_DIR` to the extracted directory, which must contain one
`<id>.jsonl` file per evaluated sample.

AAJ also needs `SEMANTIC_MANIFEST`, a task-specific JSONL with at least one
`id` field per line. Each ID must have generated audio in `AUDIO_DIR` and a
matching rubric file. The manifest itself is not included in this directory.

## Temporal file

`temporal/timestamp.jsonl` contains temporal annotations with fields `id`,
`timestamp` (`start-end` seconds), `caption`, and `type`.

* `type: "speech"` rows are used as speech-activity references for SpeechF1.
* `type: "audio"` rows describe sound-event intervals for SoundF1.

Set `TIMESTAMP_REFERENCE` to this file. SoundF1 can use an existing prediction
file through `SOUND_PREDICTION`; such a file contains `id`, `pred_timestamp`,
and `original_timestamp`. If it is empty, the pipeline can generate
predictions with `SOUND_MODEL`.

## Path quick reference

| `configs/paths.sh` variable | File or directory |
| --- | --- |
| `REF_AUDIO` | `acoustic/<split>/audio.jsonl` |
| `REF_CAPTION` | `acoustic/<split>/caption.jsonl` |
| `TEXT_REFERENCE` | `speech/transcript.jsonl` |
| `SPEAKER_PAIRS` | `speech/spk_sim.jsonl` after adapting the prompt key |
| `TIMESTAMP_REFERENCE` | `temporal/timestamp.jsonl` |
| `SEMANTIC_MANIFEST` | Task-specific AAJ manifest with `id` values |
| `RUBRIC_DIR` | Directory extracted from `semantic/aaj_rubric.zip` |
| `SOUND_PREDICTION` | Optional SoundF1 prediction JSONL |
| `AUDIO_DIR` | Generated audio directory, not a metadata file |
