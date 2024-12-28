import pyaudio
import numpy as np
from collections import deque

"""
a test to ensure the functionality of a rolling buffer with microphone input.

various tests while building a rolling buffer 

"""

RATE = 16000  # sampling rate
CHUNK = int(RATE * 0.01)  #samples per frame -> 10 ms chunks, 16000 samples per frame
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # mono
BUFFER_DURATION = 1 # length in time for the buffer
BUFFER_SIZE = RATE * BUFFER_DURATION

p = pyaudio.PyAudio()



# open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

rolling_buffer = deque(maxlen = BUFFER_SIZE)
print("buffer start...")

output_stream = p.open(format=pyaudio.paInt16,
                       channels=1,
                       rate=RATE,
                       output=True)

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        rolling_buffer.extend(audio_data)

        # test 1 -> confirm that audio data is recieved
        #print(audio_data) 

        # test 2 -> confirm that the buffer is being populated
        #print(f"Buffer length: {len(rolling_buffer)}") 

        # test 3 -> confirm that the audio being saved to the buffer is indeed audible data
        #output_stream.write(data) 




except KeyboardInterrupt:
    print("\nRecording stopped.")

finally:
    #cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
