# Fine-tune Whisper for English

This project fine-tunes the open-source `openai/whisper-small` model for English speech transcription. It is a model-training workflow, not a web app. OpenAI's hosted Whisper API does not provide a way to fine-tune its hosted model; this project trains a local Hugging Face Whisper checkpoint instead.

## Tests

Run the data-preparation unit tests with Python 3.10 or newer; they use only the Python standard library and do not download a model or dataset:

```powershell
python -m unittest discover -s tests -v
```

GitHub Actions runs these tests and checks Python syntax on Python 3.10 through 3.13 for every push and pull request. Full model training is not run in CI because it requires audio data, downloads a pretrained checkpoint, and is resource-intensive.

## 1. Prepare audio and transcripts

Training requires audio clips paired with accurate transcripts. A ready-to-start option is the English Common Voice corpus. Download and extract it, then provide the extracted directory containing `train.tsv`, `dev.tsv`, and `clips/`.

Create CSV manifests:

```powershell
python prepare_common_voice.py --corpus-dir "C:\data\cv-corpus\en" --output-dir data
```

For a small trial, add `--limit 200` to limit each split. The resulting `data\train.csv` and `data\validation.csv` contain absolute audio paths and transcripts. The helper skips missing audio clips and blank transcripts.
Common Voice clips are typically MP3; install FFmpeg and make it available on `PATH` if the audio decoder reports that it cannot read them.

For your own recordings, create two CSV files with these headers and one row per clip:

```csv
path,transcript
C:\data\audio\clip-001.wav,The exact words spoken in this recording.
```

Keep speakers or recording sessions separate between training and validation where possible. Use audio and transcripts you have permission to use. The model supports common audio formats that the installed audio decoder can read; 16 kHz mono WAV is a straightforward choice.

## 2. Install dependencies

Use Python 3.10 or newer. For GPU training, install the PyTorch build appropriate for your CUDA version from the official PyTorch installation selector first. Then install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Training is much faster with a CUDA-capable GPU. A CPU can run the workflow but may take a long time. If you run out of GPU memory, use a smaller batch size, for example `--batch-size 1`, or switch to `--model-name openai/whisper-tiny`.

## 3. Train

```powershell
python train.py --train-csv data\train.csv --validation-csv data\validation.csv --output-dir whisper-english-finetuned
```

The script evaluates word error rate (WER) after each epoch, keeps the best checkpoint, and saves the processor with the fine-tuned model. Training data must be representative of the speech you want to recognize; a small example dataset is only for testing the pipeline and will not produce a generally accurate model.

## 4. Transcribe audio

```powershell
python transcribe.py --model-dir whisper-english-finetuned --audio "C:\data\audio\new-recording.wav"
```
