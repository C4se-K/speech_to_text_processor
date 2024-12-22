import os
import time
import pyaudio 
import pathlib
import numpy as np
from collections import deque 
from faster_whisper import WhisperModel


RATE = 16000
BUFFER_LENGTH = 2.0
CAPTURE_INTERVAL = 0.5
FORMAT = pyaudio.paInt16
CHUNK = int(RATE * CAPTURE_INTERVAL)


DATASET_PATH = 'data'
data_dir = pathlib.Path(DATASET_PATH)


MODEL_SIZE = "large-v3"
rolling_buffer = deque(maxlen = int(RATE * BUFFER_LENGTH))
model = WhisperModel(MODEL_SIZE, device = "cuda", compute_type = "float16")


audio = pyaudio.PyAudio()
stream = audio.open() # todo


print("all systems nominal")


try:
    while True:
        start_time = time.time()

        audio_chunk = stream.read()


except KeyboardInterrupt:
    print('exit')
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()























