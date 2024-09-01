import config
from src.dataset.loadDataset import loadDataset
from src.database.elasticsearchConfig import connect_to_elasticsearch
from logger import logger
from src.utils.openAiGpt import create_embedding
from datetime import datetime
import PyPDF2

# Load dataset
dataset = loadDataset()

# Connect to Elasticsearch
client = connect_to_elasticsearch()


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


def populateDataToElasticsearch():
    """
        Populates data to Elasticsearch by iterating over a dataset,
        creating embeddings for each data point, and indexing them.

        Args:
            dataset (list): A list of data points to be indexed in Elasticsearch.

        Raises:
            Exception: If an error occurs during the indexing process.
    """
    try:
        for data in dataset:
            # Create request body
            embedding_str = "Quote: " + data['quote'] + " Author: " + data['author'] + " Tags: " + str(data['tags'])
            data['vector'] = create_embedding(embedding_str)
            data['createdAt'] = str(datetime.now())
            # Index data to Elasticsearch
            response = client.index(index=config.ENTITY_EMBEDDINGS_INDEX, body=data)
            logger.info(f"Data populated successfully to Elasticsearch. Response : {response}")
    except Exception as e:
        logger.error(f"An error occurred while populating data to Elasticsearch. {e}")
        raise Exception(e)


def populateTextualDataToElasticsearch():
    """
        Populates textual data to Elasticsearch by iterating over a dataset,
        creating embeddings for each data point, and indexing them.

        Args:
            dataset (list): A list of data points to be indexed in Elasticsearch.

        Raises:
            Exception: If an error occurs during the indexing process.
    """
    try:
        logger.info("Populating textual data to Elasticsearch")

        pdf_text = read_pdf("ai_doc.pdf")

        text_chunks = create_chunks(pdf_text)

        for text in text_chunks:
            # Create request body
            data = {
                "text": text,
                "vector": create_embedding(text),
                "createdAt": str(datetime.now())
            }
            # Index data to Elasticsearch
            response = client.index(index=config.TEXT_ENTITY_EMBEDDINGS_INDEX, body=data)
            logger.info(f"Data populated successfully to Elasticsearch. Response : {response}")
    except Exception as e:
        logger.error(f"An error occurred while populating data to Elasticsearch. {e}")
        raise Exception(e)


if __name__ == '__main__':
    logger.info("Populating data to Elasticsearch")
    populateTextualDataToElasticsearch()
    logger.info("Data populated successfully")
