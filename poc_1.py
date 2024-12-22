import os
import soundfile as sf
import numpy as np
import time
from scipy.signal import resample
from faster_whisper import WhisperModel

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Parameters
AUDIO_DIR = "data"  # Directory containing audio files
CHUNK_DURATION = 0.5  # Duration of each audio chunk (seconds)
MODEL_SIZE = "large-v3"  # Model size to use
TARGET_RATE = 16000  # Target sample rate

# Load Faster Whisper model
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

def resample_to_16khz(input_file, output_file):
    """Resample audio to 16kHz if needed."""
    audio, rate = sf.read(input_file)
    
    if rate != TARGET_RATE:
        print(f"Resampling from {rate} Hz to {TARGET_RATE} Hz...")
        # Calculate the new length of the audio
        num_samples = int(len(audio) * TARGET_RATE / rate)
        # Resample the audio
        audio_resampled = resample(audio, num_samples)
        # Write the resampled audio to a new file
        sf.write(output_file, audio_resampled, TARGET_RATE)
        print(f"Resampled audio saved to: {output_file}")
        return output_file
    else:
        print("Audio is already at 16kHz. No resampling needed.")
        return input_file

def simulate_streaming(file_path, chunk_duration, model):
    """Simulate real-time streaming from an audio file."""
    print(f"Simulating real-time streaming from file: {file_path}")

    # Read audio file
    audio, rate = sf.read(file_path)

    if rate != TARGET_RATE:
        raise ValueError(f"Audio file must have a 16kHz sample rate. Current rate: {rate}")
    
    # Calculate chunk size
    chunk_size = int(rate * chunk_duration)
    total_chunks = len(audio) // chunk_size

    rolling_buffer = []

    for i in range(total_chunks):
        # Get the current chunk
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        rolling_buffer.extend(chunk)

        # Convert buffer to numpy array
        buffer_array = np.array(rolling_buffer, dtype=np.float32)

        # Transcribe the buffer
        segments, _ = model.transcribe(buffer_array, beam_size=5)
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        # Mimic real-time delay
        time.sleep(chunk_duration)

def main():
    # List all audio files in the data directory
    files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav") or f.endswith(".mp3")]

    if not files:
        print(f"No audio files found in the directory: {AUDIO_DIR}")
        return

    print("Available audio files:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    # Choose a file
    choice = int(input("Select a file to stream (enter the number): ")) - 1
    if choice < 0 or choice >= len(files):
        print("Invalid choice.")
        return

    file_path = os.path.join(AUDIO_DIR, files[choice])
    resampled_file = file_path.replace(".wav", "_resampled.wav").replace(".mp3", "_resampled.wav")
    
    # Resample the file if necessary
    file_path = resample_to_16khz(file_path, resampled_file)

    # Simulate streaming
    simulate_streaming(file_path, CHUNK_DURATION, model)

if __name__ == "__main__":
    main()
