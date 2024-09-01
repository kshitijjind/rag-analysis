import pandas as pd
import PyPDF2


def read_pdf(file_path):
    # Open the PDF file
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        # Extract text from each page
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    return text


def create_chunks(text):
    # Split the text into chunks at each full stop
    # We include the full stop in the chunk by splitting on '. ' and then adding it back
    chunks = [chunk.strip() + '.' for chunk in text.split('. ') if chunk]

    return chunks


def loadDataset():
    text_chunks = create_chunks(read_pdf("src/dataset/ai_doc.pdf"))

    dataset = []
    for text in text_chunks:
        dataset.append({"text": text})

    return dataset
