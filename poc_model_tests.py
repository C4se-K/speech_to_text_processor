from faster_whisper import WhisperModel
import os
import pathlib
import time

model_size = "large-v3"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""

a test to grasp the processing time for the model.

tested 200 attempts.
beams 1-10 with ten attempts for each. -> average taken



sample test output: 

beam 1 Average Time taken for transcription: 0.97 seconds. time per second: 0.03 with range of: 0.0681        
beam 2 Average Time taken for transcription: 1.04 seconds. time per second: 0.03 with range of: 0.0307        
beam 3 Average Time taken for transcription: 1.11 seconds. time per second: 0.03 with range of: 0.0921
beam 4 Average Time taken for transcription: 1.76 seconds. time per second: 0.05 with range of: 0.0850
beam 5 Average Time taken for transcription: 1.82 seconds. time per second: 0.05 with range of: 0.0526
beam 6 Average Time taken for transcription: 1.97 seconds. time per second: 0.06 with range of: 0.1039
beam 7 Average Time taken for transcription: 2.05 seconds. time per second: 0.06 with range of: 0.1422
beam 8 Average Time taken for transcription: 1.95 seconds. time per second: 0.06 with range of: 0.0751
beam 9 Average Time taken for transcription: 2.25 seconds. time per second: 0.07 with range of: 0.0908
beam 10 Average Time taken for transcription: 2.51 seconds. time per second: 0.07 with range of: 0.2888
beam 11 Average Time taken for transcription: 2.53 seconds. time per second: 0.07 with range of: 0.2479
beam 12 Average Time taken for transcription: 2.64 seconds. time per second: 0.08 with range of: 0.0991
beam 13 Average Time taken for transcription: 2.80 seconds. time per second: 0.08 with range of: 0.1175
beam 14 Average Time taken for transcription: 3.01 seconds. time per second: 0.09 with range of: 0.2720
beam 15 Average Time taken for transcription: 3.24 seconds. time per second: 0.10 with range of: 0.1895
beam 16 Average Time taken for transcription: 2.94 seconds. time per second: 0.09 with range of: 0.1385
beam 17 Average Time taken for transcription: 3.49 seconds. time per second: 0.10 with range of: 0.2092
beam 18 Average Time taken for transcription: 3.74 seconds. time per second: 0.11 with range of: 0.1914
beam 19 Average Time taken for transcription: 3.84 seconds. time per second: 0.11 with range of: 0.1641
beam 20 Average Time taken for transcription: 3.98 seconds. time per second: 0.12 with range of: 0.0758


"""

# run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# measure the time to transcribe
def transcribe(i):
    start_time = time.time()  # Start timing

    segments, _ = model.transcribe(PATH, beam_size= i)

    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        if do_print:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    end_time = time.time()  # End timing

    # calculate and print time taken
    time_taken = end_time - start_time
    #print(f"beam {i} Time taken for transcription: {time_taken:.2f} seconds. time per second: {(time_taken / 34 ):.2f}")
    return time_taken


PATH = pathlib.Path("data/doorstuck.wav")
do_print = False

#take note: preloading a dummy makes the first inference faster
#otherwise, chunking works
segments, _ = model.transcribe(PATH, beam_size= 1)

#it seems there is a limit to the overhead of beam?

#200 tests in total
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