from faster_whisper import WhisperModel
import os
import pathlib
import time  # Import the time module

model_size = "large-v3"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Measure the time to transcribe
def transcribe(i):
    start_time = time.time()  # Start timing

    segments, _ = model.transcribe(PATH, beam_size= i)

    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        if do_print:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    end_time = time.time()  # End timing

    # Calculate and print time taken
    time_taken = end_time - start_time
    #print(f"beam {i} Time taken for transcription: {time_taken:.2f} seconds. time per second: {(time_taken / 34 ):.2f}")
    return time_taken


PATH = pathlib.Path("data/doorstuck.wav")
do_print = False

#take note: preloading a dummy makes the first inference faster
#otherwise, chunking works
segments, _ = model.transcribe(PATH, beam_size= 1)

#it seems there is a limit to the overhead of beam?


for i in range(1, 21):
    total = 0
    high =0
    low = 0
    for n in range(10):
        val = transcribe(i)
        total += val

        # testing for times
        if n == 0:
            high = val
            low = val

        if val > high:
            high = val
        elif val < low:
            low = val

    print(f"beam {i} Average Time taken for transcription: {(total / 10):.2f} seconds. time per second: {(total / 10 / 34 ):.2f} with range of: {(high - low):.4f}")