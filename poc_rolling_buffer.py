import pyaudio
import numpy as np
from collections import deque

"""
a test to ensure the functionality of a rolling buffer with microphone input.

various tests while building a rolling buffer 

"""


CHUNK = 1024  #samples per frame
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # mono
RATE = 44100  # sampling rate

p = pyaudio.PyAudio()

BUFFER_SIZE = 10 * CHUNK

# open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

rolling_buffer = deque(maxlen = BUFFER_SIZE)
print("buffer start...")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)

        audio_data = np.frombuffer(data, dtype=np.int16)

        rolling_buffer.extend(audio_data)

        #print(audio_data) # test 1 -> confirm that audio data is recieved
        print(f"Buffer length: {len(rolling_buffer)}") # test 2 -> confirm that the buffer is being populated

except KeyboardInterrupt:
    print("\nRecording stopped.")

finally:
    #cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
