import argparse
from pathlib import Path

import torch
from transformers import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with a fine-tuned Whisper model.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        parser.error(f"Model directory not found: {args.model_dir}")
    if not args.audio.is_file():
        parser.error(f"Audio file not found: {args.audio}")

    transcriber = pipeline(
        "automatic-speech-recognition",
        model=str(args.model_dir),
        device=0 if torch.cuda.is_available() else -1,
    )
    result = transcriber(str(args.audio), generate_kwargs={"language": "english", "task": "transcribe"})
    print(result["text"])


if __name__ == "__main__":
    main()
