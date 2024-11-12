import pyaudio
import json
from vosk import Model, KaldiRecognizer

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Load the model
model = Model(r".\vosk-model-en-us-0.42-gigaspeech")
recognizer = KaldiRecognizer(model, RATE)

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
stream.start_stream()

print("Listening...")

try:
    while True:
        data = stream.read(CHUNK)
        
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            text = json.loads(result).get('text', '')
            if text:
                print(f"Text: {text}")
        else:
            partial_result = recognizer.PartialResult()
            partial_text = json.loads(partial_result).get('partial', '')
            if partial_text:
                print(f"Partial Text: {partial_text}", end='\r')

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
