#!/usr/bin/env python3
"""Unified runner for overall, voice-prompt, timestamp and custom evaluation."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [sys.executable, str(ROOT / script), *map(str, args)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def require(ns, *names):
    missing = [name for name in names if not getattr(ns, name)]
    if missing:
        raise SystemExit("Missing arguments: " + ", ".join("--" + n.replace("_", "-") for n in missing))


def distribution(ns, out):
    require(ns, "audio_dir", "ref_audio")
    args = ["--ref_audio_jsonl", ns.ref_audio, "--gen_audio_dir", ns.audio_dir,
            "--output_file", out / "distribution.txt", "--num_workers", ns.num_workers]
    if ns.ref_caption:
        args += ["--ref_caption_jsonl", ns.ref_caption]
    if ns.recalculate: args.append("--recalculate")
    run("metrics/acoustic/t2a.py", *args)


def perception(ns, out):
    require(ns, "audio_dir", "audiobox_model")
    run("metrics/acoustic/audiobox.py", "--input-dir", ns.audio_dir,
        "--output", out / "audiobox.jsonl", "--model", ns.audiobox_model)


def speech(ns, selected, out):
    require(ns, "audio_dir")
    if "wer" in selected:
        require(ns, "text_reference", "whisper_model")
        trans = out / "transcription.jsonl"
        run("metrics/speech/whisper_asr.py", "--audio-dir", ns.audio_dir, "--manifest", ns.text_reference,
            "--output", trans, "--model", ns.whisper_model, "--device", ns.device)
        run("metrics/speech/wer.py", "--reference", ns.text_reference, "--prediction", trans,
            "--output", out / "wer.jsonl", "--language", ns.language)
    if "speaker-sim" in selected:
        require(ns, "speaker_pairs")
        run("metrics/speech/speaker_similarity.py", "--pairs", ns.speaker_pairs,
            "--output", out / "speaker_sim.jsonl", "--device", ns.device)
    if "utmos" in selected:
        args = [ns.audio_dir, out / "utmos.jsonl"]
        if ns.utmos_checkpoint: args += ["--checkpoint", ns.utmos_checkpoint]
        run("metrics/speech/utmos.py", *args)


def semantic(ns, selected, out):
    require(ns, "audio_dir")
    if "clap" in selected:
        require(ns, "ref_caption", "clap_model")
        run("metrics/semantic/clap.py", "--audio-dir", ns.audio_dir, "--captions", ns.ref_caption,
            "--output", out / "clap.jsonl", "--model", ns.clap_model)
    if "aaj" in selected:
        require(ns, "rubric_dir", "judge_model")
        require(ns, "semantic_manifest")
        run("metrics/semantic/aaj.py", "--audio-dir", ns.audio_dir, "--manifest", ns.semantic_manifest,
            "--rubric-dir", ns.rubric_dir, "--model", ns.judge_model, "--output-dir", out / "aaj")


def temporal(ns, selected, out):
    require(ns, "audio_dir", "timestamp_reference")
    if "speech_f1" in selected:
        require(ns, "whisperx_model")
        pred = out / "speech_segments.jsonl"
        run("metrics/temporal/whisperx_timestamp.py", "--audio-dir", ns.audio_dir, "--manifest", ns.timestamp_reference,
            "--output", pred, "--model", ns.whisperx_model, "--device", ns.device)
        run("metrics/temporal/speech_f1.py", "--prediction", pred, "--reference", ns.timestamp_reference,
            "--output", out / "speech_f1.jsonl")
    if "sound_f1" in selected:
        sound_prediction = ns.sound_prediction
        if not sound_prediction and ns.sound_model:
            sound_prediction = out / "sound_segments.jsonl"
            run("metrics/temporal/pe_a_frame.py", "--audio-dir", ns.audio_dir,
                "--manifest", ns.timestamp_reference, "--output", sound_prediction,
                "--model", ns.sound_model, "--threshold", ns.sound_threshold)
        if not sound_prediction:
            require(ns, "sound_prediction")
        run("metrics/temporal/sound_f1.py", "--input", sound_prediction,
            "--output", out / "sound_f1.jsonl")


def make_parser():
    p = argparse.ArgumentParser(description="Audio benchmark evaluation")
    p.add_argument("mode", choices=["overall", "voice-prompt", "timestamp", "custom"])
    p.add_argument("--audio-dir"); p.add_argument("--output-dir")
    p.add_argument("--ref-audio"); p.add_argument("--ref-caption")
    p.add_argument("--text-reference"); p.add_argument("--whisper-model")
    p.add_argument("--speaker-pairs"); p.add_argument("--whisperx-model"); p.add_argument("--timestamp-reference")
    p.add_argument("--sound-prediction"); p.add_argument("--sound-model"); p.add_argument("--sound-threshold", type=float, default=0.3)
    p.add_argument("--semantic-manifest"); p.add_argument("--rubric-dir"); p.add_argument("--judge-model")
    p.add_argument("--audiobox-model"); p.add_argument("--utmos-checkpoint"); p.add_argument("--clap-model")
    p.add_argument("--device", default="cuda"); p.add_argument("--language", choices=["en", "zh"], default="en")
    p.add_argument("--num-workers", type=int, default=4); p.add_argument("--recalculate", action="store_true")
    p.add_argument("--metrics", nargs="+", choices=["acoustic", "speech", "semantic", "temporal",
        "distribution", "perception", "wer", "speaker-sim", "utmos", "clap", "aaj", "sound_f1", "speech_f1"])
    return p


def main():
    ns = make_parser().parse_args()
    if ns.mode == "overall":
        selected = {"distribution", "perception", "wer", "utmos", "clap", "aaj"}
    elif ns.mode == "voice-prompt":
        selected = {"distribution", "perception", "wer", "speaker-sim", "utmos", "clap", "aaj"}
    elif ns.mode == "timestamp":
        selected = {"distribution", "perception", "wer", "utmos", "clap", "aaj", "sound_f1", "speech_f1"}
    else:
        selected = set(ns.metrics or [])
    require(ns, "output_dir")
    groups = {
        "acoustic": {"distribution", "perception"},
        "speech": {"wer", "speaker-sim", "utmos"},
        "semantic": {"clap", "aaj"},
        "temporal": {"sound_f1", "speech_f1"},
    }
    for group, metrics in groups.items():
        if group in selected:
            selected.update(metrics)
    out = Path(ns.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if "distribution" in selected: distribution(ns, out)
    if "perception" in selected: perception(ns, out)
    if selected & {"wer", "speaker-sim", "utmos"}:
        speech(ns, selected, out)
    if selected & {"clap", "aaj"}:
        semantic(ns, selected, out)
    if selected & {"sound_f1", "speech_f1"}:
        temporal(ns, selected, out)


if __name__ == "__main__":
    main()
