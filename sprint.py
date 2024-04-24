import wave

import numpy as np
import scipy
import torch
from pydub import AudioSegment
from transformers import AutoModelForTextToWaveform, AutoTokenizer, VitsModel


def generate_speech(text):
    model_name = "facebook/mms-tts-deu"  # Replace with the model you selected
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = VitsModel.from_pretrained(model_name)
    model = AutoModelForTextToWaveform.from_pretrained(model_name)
    # Load model directly

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    output = output.squeeze().cpu().numpy().astype("float32")

    # Normalize the audio data to the range [-1, 1]
    output /= np.max(np.abs(output))
    # Convert waveform data to a compatible data type (e.g., int16)
    #    output = output.squeeze().cpu().numpy().astype("int16")

    # Write the waveform to a WAV file
    scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output)


#    scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output)
# Save the audio


#    with wave.open("output.wav", "wb") as audio_file:
#        audio_file.setparams((1, 2, 22050, 0, "NONE", "NONE"))
#        audio_file.writeframes(output.numpy().tobytes())


# Example usage
text_to_speak = "Meine Schwester Elena stinkt"
generate_speech(text_to_speak)
