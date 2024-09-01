import config
from src.database.elasticsearchConfig import connect_to_elasticsearch
from logger import logger
from src.dataset.loadDataset import loadDataset
from src.utils.openAiGpt import create_embedding
from datetime import datetime

# Connect to Elasticsearch
client = connect_to_elasticsearch()


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

        text_chunks = loadDataset()

        for text in text_chunks:
            # Create request body
            data = {
                "text": text['text'],
                "vector": create_embedding(text['text']),
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
