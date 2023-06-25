import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import torch
from glob import glob

device = torch.device('cpu')  # 'cuda' can also be used for GPU acceleration
model, decoder, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_stt',
    language='en',
    device=device
)
(read_batch, split_into_batches, read_audio, prepare_model_input) = utils

# Set the sample rate and duration for recording
sample_rate = 44100  # Typically used sample rate
duration = 5  # Duration in seconds

# List to store the recorded audio data
audio_data = []

# Callback function to handle the recorded audio data
def audio_callback(indata, frames, time, status):
    # Append the recorded audio data to the list
    audio_data.append(indata.copy())


while True:
    try:
        # Start recording with the specified settings and callback function
        with sd.InputStream(callback=audio_callback, channels=2, samplerate=sample_rate):
            sd.sleep(int(duration * 1000))  # Wait for the specified duration in milliseconds
        # Convert the recorded audio data to a numpy array
        audio_array = np.concatenate(audio_data)
        # Save the audio array to a .wav file
        sf.write('output_live.wav', audio_array, sample_rate)

        test_files = glob('output_live.wav')
        batches = split_into_batches(test_files, batch_size=10)
        input_data = prepare_model_input(read_batch(batches[0]), device=device)

        output = model(input_data)
        for transcription in output:
            decoded_text = decoder(transcription.cpu())
            print(decoded_text)

        # Delete the temporary .wav file
        os.remove('output_live.wav')
        # Dump the audio array to prevent keeping fed content (this can be kept in order to 1 long S2T)
        audio_data = []

    except KeyboardInterrupt:
        # User interrupted the loop
        print('-------------------------')
        print("Loop interrupted by user.")
        print('-------------------------')
        break
