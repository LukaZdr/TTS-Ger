import os
import time

import numpy as np
import scipy
import torch
from pypdf import PdfReader
from transformers import AutoModelForTextToWaveform, AutoTokenizer, VitsModel


def generate_speech(text, model, tokenizer, file_name="test.wav"):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    output = output.squeeze().cpu().numpy().astype("float32")

    # Normalize the audio data to the range [-1, 1]
    output /= np.max(np.abs(output))

    # Write the waveform to a WAV file
    scipy.io.wavfile.write(file_name, rate=model.config.sampling_rate, data=output)


large_text = ""
# list all files in the "current_pdf" dir
pdf_dir = "current_pdf"
files = os.listdir(pdf_dir)
pdf_name = files[0]
reader = PdfReader(pdf_dir + "/" + pdf_name)
model_name = "facebook/mms-tts-deu"  # Replace with the model you selected
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTextToWaveform.from_pretrained(model_name)

for page in reader.pages[3:91]:
    print(f"Start Page {page.page_number}")
    start_time = time.time()
    generate_speech(
        page.extract_text(),
        model,
        tokenizer,
        f"output/{pdf_name[:-4]}_{page.page_number}.wav",
    )
    print(f"Page {page.page_number} took {time.time()-start_time} seconds")
