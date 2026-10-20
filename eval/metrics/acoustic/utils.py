import json
from typing import Union, Dict
from pathlib import Path
import os

from omegaconf import OmegaConf



def read_jsonl_to_mapping(
    jsonl_file: Union[str, Path],
    key_col: str,
    value_col: str,
    base_path=None
) -> Dict[str, str]:
    """
    Read two columns, indicated by `key_col` and `value_col`, from the
    given jsonl file to return the mapping dict
    TODO handle duplicate keys
    """
    mapping = {}
    with open(jsonl_file, 'r') as file:
        for line in file.readlines():
            data = json.loads(line.strip())
            #print(data, flush = True)
            key = data[key_col]
            value = data[value_col]
            if base_path:
                value = os.path.join(base_path, value)
            mapping[key] = value

    #print(mapping, flush=True)
    return mapping

def audio_dir_to_mapping(audio_dir: str | Path, task: str) -> dict:
    mapping = {}
    audio_dir = Path(audio_dir)
    audio_files = sorted(audio_dir.iterdir())
    for audio_file in audio_files:
        audio_id = audio_file.stem
        mapping[audio_id] = str(audio_file.resolve())
    return mapping
