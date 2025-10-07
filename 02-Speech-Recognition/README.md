# Speech Recognition

Offline/online speech-to-text system that transcribes audio files into text.

## Setup
- Python 3.9+
- Install dependencies from repo root:
  - `pip install -r ../requirements.txt` (from this folder)
  - or `pip install -r requirements.txt` (from repo root)

## Input
- `sample_audio.wav` — example WAV file
- Other formats may be supported depending on the backend (`pydub/ffmpeg` required for conversions)

## Usage
Run from this folder:

```bash
python speech_recognition_system.py --audio sample_audio.wav --engine google
```

Flags:
- `--audio PATH` — path to input audio file
- `--engine {google,sphinx,whisper}` — choose backend if supported by the script
- `--language CODE` — language code (e.g., `en-US`)

If no flags are provided, the script uses `sample_audio.wav` if present.

## Outputs
- `result.txt` — transcribed text
- `output_speech _recog.png` — visualization/summary of transcription (filename preserved as in repo)
- `speech _recoz.png` — additional graphic output (filename preserved)

## Examples
```bash
# Local CMU Sphinx backend (if available)
python speech_recognition_system.py --audio sample_audio.wav --engine sphinx

# OpenAI Whisper (requires separate install/API if applicable)
python speech_recognition_system.py --audio sample_audio.wav --engine whisper --language en
```

## Notes
- Some engines require credentials or additional installs (e.g., ffmpeg, API keys). See `requirements.txt` and engine docs.
