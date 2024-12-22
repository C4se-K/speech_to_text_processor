import pyaudio
import numpy as np

# Constants
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate in Hz

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording. Press Ctrl+C to stop.")

try:
    while True:
        # Read audio data from the stream
        data = stream.read(CHUNK, exception_on_overflow=False)

        # Convert binary data to NumPy array
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Print the audio data
        print(audio_data)

except KeyboardInterrupt:
    print("\nRecording stopped.")

finally:
    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()
