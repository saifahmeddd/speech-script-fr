import whisper
import sounddevice as sd
import numpy as np
import torch
import tempfile
import os
import warnings
from scipy.io.wavfile import write
from deep_translator import GoogleTranslator
import time

warnings.filterwarnings("ignore", category=UserWarning)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium").to(device)

samplerate = 16000
duration = 7  
channels = 1

print("\nspeak in English or French.\n")
print("⏳ Preparing microphone...")
sd.rec(1, samplerate=samplerate, channels=channels)
sd.wait()
print("Ready!\n")

try:
    while True:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='float32')
        sd.wait()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            write(tmpfile.name, samplerate, recording)
            audio_path = tmpfile.name
        result = model.transcribe(audio_path)
        detected_lang = result.get("language", "en")
        text = result["text"].strip()

        if text:
            if detected_lang == "en":
                translated = GoogleTranslator(source="en", target="fr").translate(text)
                print(translated, "\n")
            elif detected_lang == "fr":
                translated = GoogleTranslator(source="fr", target="en").translate(text)
                print("français:", text)
                print("english:", translated, "\n")
            else:
                print(f"detected unsupported language: {detected_lang}")

        os.remove(audio_path)

except KeyboardInterrupt:
    print("\ntranslation stopped. Goodbye!")
