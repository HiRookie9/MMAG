#!/usr/bin/env python3
"""
Text-to-Audio acoustic evaluation script.

This script evaluates generated audio against reference audio using acoustic
distribution metrics such as FAD, FD, KL and IS. CLAP is implemented separately
in ``metrics/semantic/clap.py``.
"""

import argparse
from collections import defaultdict
import os
from copy import deepcopy
from pathlib import Path

import torch

# Ref: https://github.com/haoheliu/audioldm_eval/tree/main
# This script uses a locally modified version of audioldm_eval.
from audioldm_eval import EvaluationHelper

from utils import read_jsonl_to_mapping, audio_dir_to_mapping

torch.multiprocessing.set_sharing_strategy('file_system')


def get_common_folder_path(audio_dict):
    """
    Extract the common folder path from audio path dictionary.
    
    Parameters:
    audio_dict -- Dictionary in format {audio_id: audio_path}
    
    Returns:
    common_folder -- Common folder path (None if no common path)
    is_same_folder -- Boolean indicating if all audios are in the same folder
    """
    if not audio_dict:
        return None, False
    paths = list(audio_dict.values())
    parent_folders = [os.path.dirname(path) for path in paths]
    # print("parent_folders:", parent_folders, flush=True)  # Debug only.
    common_prefix = str(Path(os.path.commonpath(parent_folders)).resolve())
    is_same_folder = all(
        parent == parent_folders[0] for parent in parent_folders
    )

    return common_prefix, is_same_folder


def evaluate(args):
    """Calculate FAD, FD, KL, etc. socres."""
    #print(args.ref_audio_jsonl, flush=True)
    ref_aid_to_audios = read_jsonl_to_mapping(
        args.ref_audio_jsonl,
        "audio_id",
        "audio",
    )
    if args.gen_audio_jsonl is not None:
        gen_aid_to_audios = read_jsonl_to_mapping(
            args.gen_audio_jsonl, "audio_id", "audio"
        )
    elif args.gen_audio_dir is not None:
        gen_aid_to_audios = audio_dir_to_mapping(args.gen_audio_dir, args.task)

    keys = deepcopy(list(ref_aid_to_audios.keys()))
    for key in keys:
        if key not in gen_aid_to_audios:
            ref_aid_to_audios.pop(key)
    """Calculate ldm eval score: FAD, FD, KL score"""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = "cnn14" if args.task == "tta" else "mert"
    evaluator = EvaluationHelper(16000, args.device, backbone=backbone)
    #print(ref_aid_to_audios, flush=True)
    gen_folder_path, gen_is_same_folder = get_common_folder_path(
        gen_aid_to_audios
    )
    ref_folder_path, ref_is_same_folder = get_common_folder_path(
        ref_aid_to_audios
    )
    # print(gen_folder_path, flush=True)
    # print(ref_folder_path, flush=True)
    assert gen_is_same_folder == True, "Generated audio files must be in the same folder."
    assert ref_is_same_folder == True, "Reference audio files must be in the same folder."

    eval_result = evaluator.main(
        gen_aid_to_audios,
        ref_aid_to_audios,
        recalculate=args.recalculate,
        num_workers=args.num_workers,
    )

    assert ref_aid_to_audios.keys() == gen_aid_to_audios.keys(
    ), "Reference and generated audio IDs do not match"

    results = defaultdict(dict)
    results.update(eval_result)
    with open(args.output_file, "w") as writer:
        for metric, values in results.items():
            print_msg = f"{metric}: {values:.3f}"
            print(print_msg)
            print(print_msg, file=writer)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=
        "Evaluate text-to-audio generation using acoustic distribution metrics.",
        epilog="""
Examples:
    python metrics/acoustic/t2a.py \\
      --ref_audio_jsonl data/audiocaps_v2/test/audio.jsonl \\
      --rc data/audiocaps_v2/test/caption.jsonl \\
      --gd xxxx/tta_infer \\
      -o xxx/tta.txt
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ref_audio_jsonl",
        "-r",
        type=str,
        required=True,
        help="path to reference audio jsonl file"
    )
    parser.add_argument(
        "--ref_caption_jsonl",
        "-rc",
        type=str,
        required=False,
        help="path to reference caption jsonl file"
    )
    parser.add_argument(
        "--gen_audio_dir",
        "-gd",
        type=str,
        help="path to generated audio directory"
    )
    parser.add_argument(
        "--gen_audio_jsonl",
        "-gj",
        type=str,
        help="path to generated audio jsonl file"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        required=True,
        help="path to output file"
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="tta",
        help="task type, text-to-audio (tta) or text_to_music (ttm)",
        choices=["tta", "ttm"]
    )
    parser.add_argument(
        "--num_workers",
        "-c",
        default=4,
        type=int,
        help="number of workers for parallel processing"
    )
    parser.add_argument(
        "--recalculate",
        action="store_true",
        help="recalculate embeddings for metric scores"
    )
    args = parser.parse_args()

    evaluate(args)
