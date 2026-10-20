import json
import string
import argparse
from typing import Dict, List
from jiwer import process_words, wer
from whisper_normalizer.english import EnglishTextNormalizer
from zhon.hanzi import punctuation

# Initialize the English normalizer.
english_normalizer = EnglishTextNormalizer()

def clean_for_wer(text: str, lang: str = "en") -> str:
    """
    Normalize text before WER calculation.
    """
    # Remove ASCII and Chinese punctuation.
    for x in string.punctuation + punctuation:
        text = text.replace(x, "")
    # Remove repeated spaces.
    text = text.replace("  ", " ")
    if lang == "zh":
        # Separate Chinese characters for tokenization.
        text = " ".join([x for x in text])
    else:
        # Apply Whisper normalization for English text.
        text = english_normalizer(text)
        text = text.lower()
    return text.strip()

def load_refs(ref_file: str, lang: str = "en") -> Dict[str, str]:
    """
    Load a reference file and return id -> normalized text.
    """
    ref_dict = {}
    with open(ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            id_ = obj.get("id")
            trans = obj.get("transcription", obj.get("text", ""))
            if id_ is not None and trans:
                ref_dict[str(id_)] = clean_for_wer(trans, lang=lang)
    return ref_dict

def compute_wer_for_file(pred_file: str, ref_dict: Dict[str, str], lang: str = "en") -> List[Dict]:
    """
    Compute one WER value per prediction.
    """
    results = []
    total_wer = 0.0
    count = 0
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            id_ = obj.get("id")
            if id_ is None:
                continue
            pred_text = obj.get("text", obj.get("transcription", ""))
            if not pred_text:
                continue

            # Look up the already-normalized reference.
            ref_clean = ref_dict.get(str(id_))
            if ref_clean is None:
                print(f"Warning: id {id_} is missing from the reference; skipping")
                continue

            # Normalize the prediction.
            pred_clean = clean_for_wer(pred_text, lang=lang)

            # Compute WER.
            error = wer(ref_clean, pred_clean)
            error = min(1.0, error)
            # Store the detailed result.
            results.append({
                "id": id_,
                "wer": error,
                "ref_text": ref_clean,
                "pred_text": pred_clean
            })
            total_wer += error
            count += 1

    avg_wer = total_wer / count if count else 0.0
    print(f"Samples: {count}, mean WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")

    results.append({
                    "pred_file": pred_file,
                    "avg_wer": f"{avg_wer:.4f}",
                })
    return results

def main(ref_path: str, pred_path: str, output_path: str, lang: str = "en"):
    # Load normalized references.
    ref_dict = load_refs(ref_path, lang=lang)
    # Compute detailed per-sample WER values.
    results = compute_wer_for_file(pred_path, ref_dict, lang=lang)
    # Write one JSON object per line.
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Detailed results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute WER from reference and prediction JSONL files.")
    parser.add_argument("--reference", required=True, help="JSONL with id and text/transcription")
    parser.add_argument("--prediction", required=True, help="JSONL with id and text/transcription")
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    args = parser.parse_args()
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    main(args.reference, args.prediction, args.output, args.language)
