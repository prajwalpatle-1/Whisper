# Whisper fine-tuning for English transcription

This project fine-tunes the pretrained Whisper model for a specific English speech transcription task.

## What we are doing

We are taking the open-source `openai/whisper-small` checkpoint and adapting it to a custom English ASR dataset. The goal is not to build a web app or an external API, but to train a local speech recognition model that performs better on the target speech domain.

This is a standard transfer-learning workflow:

- start from a general Whisper checkpoint
- prepare audio/transcript pairs
- fine-tune the model on English examples
- save the updated model for local inference

## Model understanding

Whisper is a multilingual speech recognition model built as an encoder-decoder transformer.

The workflow is:

- audio is converted into a spectrogram-like representation
- the encoder extracts speech features from that input
- the decoder generates text tokens conditioned on the audio features
- the model is trained to produce transcripts in a target language/task setting

For this project, we explicitly set:

- `language = "english"`
- `task = "transcribe"`

That matters because Whisper can do more than transcription. It can also be used for multilingual transcription and translation, but here we are focusing on English transcription only.

## Why `openai/whisper-small`

The default base model is:

- `openai/whisper-small`

This is a good compromise between performance and training cost. It is larger than `tiny`, but much lighter than the larger Whisper variants. It gives enough capacity to learn a target English accent, domain, or recording style while remaining practical for local fine-tuning.

## What the training does

The project loads the pretrained Whisper checkpoint and fine-tunes it using labeled audio examples.

At a high level:

- each audio clip is converted into Whisper-compatible features
- the transcript is tokenized as the target sequence
- the model learns to predict the transcript from the audio input
- the trainer evaluates word error rate (WER) after each epoch
- the best model checkpoint is kept and saved

This makes the model more specialized for the speech data we care about, instead of relying only on the generic pretrained version.

## Output of the project

After training, the project produces a local fine-tuned Whisper model directory that can be used to transcribe new English audio files with the model-specific pipeline.

The final behavior is:

- load saved model checkpoint
- pass audio to Whisper
- generate English transcript text

## Summary

This repository is a model adaptation project: we are taking a pretrained Whisper model and fine-tuning it for English transcription on a specific dataset or domain. The core focus is on Whisper as a speech-to-text model and how we specialize it through supervised fine-tuning.
