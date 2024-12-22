import pyaudio
import numpy as np

"""
a test to ensure the functionality of a rolling buffer with microphone input.



"""



# Constants
CHUNK = 1024  #samples per frame
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # mono
RATE = 44100  # sampling rate

p = pyaudio.PyAudio()

# open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)

        audio_data = np.frombuffer(data, dtype=np.int16)

        print(audio_data)

except KeyboardInterrupt:
    print("\nRecording stopped.")

finally:
    #cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
