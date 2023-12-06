from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
import whisper
import torch
import ffmpeg
from typing import List
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("hello")

# Checking if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)


#model paths : https://github.com/openai/whisper/discussions/734

# Load the Whisper model:
model_dir = "whisper-tiny-openai.pt"
#model_dir = "/Users/abhijeetrajput/myfiles/whisper_tiny_en/tiny.en.pt"
#"/Users/abhijeetrajput/myfiles/whisper_large_v2"
model = whisper.load_model(model_dir)


# Print some model details
print(model.parameters())
print(model.is_multilingual)

audio_input = "/Users/abhijeetrajput/myfiles/whisper_tiny_en/tinlin-headphones-74657.mp3"
#transcription = model.transcribe(audio_input)

import sys
import os
sys.path.append('/opt/homebrew/bin/ffmpeg')
os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin/'

audio = whisper.load_audio(audio_input)
audio = whisper.pad_or_trim(audio)
print(model.transcribe(audio)['text'])