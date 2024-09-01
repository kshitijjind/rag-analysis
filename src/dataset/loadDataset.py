import PyPDF2
import os

file = os.path.join(os.path.dirname(os.getcwd()), "rag-analysis/src/dataset/ai_doc.pdf")


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
    # Create text chunks of 500 words
    text_chunks = []

    words = text.split()
    chunk = ""
    for word in words:
        if len(chunk.split()) < 250:
            chunk += word + " "
        else:
            text_chunks.append(chunk.strip())
            chunk = word + " "

    # Add the last chunk if it contains any words
    if chunk:
        text_chunks.append(chunk.strip())

    return text_chunks


def loadDataset():
    text_chunks = create_chunks(read_pdf(file))

    dataset = []
    for text in text_chunks:
        dataset.append({"text": text})

    return dataset
