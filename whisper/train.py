import argparse
import csv
from pathlib import Path

import torch
from datasets import Audio, Dataset
from jiwer import wer
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def load_manifest(csv_path: Path) -> Dataset:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"path", "transcript"}.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain 'path' and 'transcript' columns")

        for row in reader:
            audio_path = Path(row["path"])
            if not audio_path.is_absolute():
                audio_path = csv_path.parent / audio_path
            transcript = row["transcript"].strip()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file listed in {csv_path} was not found: {audio_path}")
            if transcript:
                rows.append({"path": str(audio_path.resolve()), "transcript": transcript})

    if not rows:
        raise ValueError(f"No usable audio/transcript rows found in {csv_path}")

    return Dataset.from_list(rows).cast_column("path", Audio(sampling_rate=16_000))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for English transcription.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("whisper-english-finetuned"))
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("--epochs, --batch-size, and --learning-rate must be positive")

    processor = WhisperProcessor.from_pretrained(
        args.model_name, language="English", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    train_dataset = load_manifest(args.train_csv)
    validation_dataset = load_manifest(args.validation_csv)

    def prepare_example(example):
        audio = example["path"]
        features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        labels = processor.tokenizer(example["transcript"]).input_ids
        return {"input_features": features, "labels": labels}

    train_dataset = train_dataset.map(
        prepare_example, remove_columns=train_dataset.column_names
    )
    validation_dataset = validation_dataset.map(
        prepare_example, remove_columns=validation_dataset.column_names
    )

    def collate_features(features):
        input_features = [{"input_features": item["input_features"]} for item in features]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": item["labels"]} for item in features]
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        if (labels[:, 0] == model.config.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

    def compute_metrics(prediction):
        predicted_ids = prediction.predictions
        if isinstance(predicted_ids, tuple):
            predicted_ids = predicted_ids[0]
        label_ids = prediction.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predicted_text = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        label_text = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer(label_text, predicted_text)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_steps=50,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate_features,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"Fine-tuned model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
