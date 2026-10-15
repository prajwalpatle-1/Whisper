import argparse
import csv
from pathlib import Path


def write_manifest(tsv_path: Path, clips_dir: Path, output_path: Path, limit: int | None) -> int:
    clips_dir = clips_dir.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with tsv_path.open("r", encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source, delimiter="\t")
        if not rows.fieldnames or not {"path", "sentence"}.issubset(rows.fieldnames):
            raise ValueError(f"{tsv_path} must contain 'path' and 'sentence' columns")

        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=["path", "transcript"])
            writer.writeheader()

            for row in rows:
                clip_path = (clips_dir / row["path"]).resolve()
                if not clip_path.is_relative_to(clips_dir):
                    continue
                transcript = row["sentence"].strip()
                if not transcript or not clip_path.is_file():
                    continue

                writer.writerow({"path": str(clip_path), "transcript": transcript})
                written += 1
                if limit is not None and written >= limit:
                    break

    if written == 0:
        raise ValueError(f"No usable audio/transcript rows found in {tsv_path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Whisper CSV manifests from an extracted Common Voice English corpus."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Extracted corpus directory containing train.tsv, dev.tsv, and clips/",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, help="Optional maximum rows per split")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    clips_dir = args.corpus_dir / "clips"
    if not clips_dir.is_dir():
        parser.error(f"Audio directory not found: {clips_dir}")

    for split, tsv_name in (("train", "train.tsv"), ("validation", "dev.tsv")):
        tsv_path = args.corpus_dir / tsv_name
        if not tsv_path.is_file():
            parser.error(f"Split file not found: {tsv_path}")
        count = write_manifest(
            tsv_path, clips_dir, args.output_dir / f"{split}.csv", args.limit
        )
        print(f"Wrote {count} rows to {args.output_dir / f'{split}.csv'}")


if __name__ == "__main__":
    main()
