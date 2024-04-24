import PyPDF2
import scipy
import torch
from transformers import AutoTokenizer, VitsModel

large_text = ""
# with open("sample.pdf", "rb") as pdf_file:
#    read_pdf = PyPDF2.PdfFileReader(pdf_file)
#    number_of_pages = read_pdf.getNumPages()
#    for page in read_pdf.pages:
#        large_text = "/n" + page.extractText()
#
model = VitsModel.from_pretrained("facebook/mms-tts-deu")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-deu")

large_text = "Das hier ist ein test satz"
inputs = tokenizer(large_text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output)
