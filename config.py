import os


def get_environment_variable(key, default_value):
    return os.environ.get(key, default_value)


# Logging
LOG_LEVEL = get_environment_variable('LOG_LEVEL', 'INFO')

# OpenSearch
ELASTICSEARCH_DOMAIN_URL = os.getenv('ELASTICSEARCH_DOMAIN_URL','localhost')
ELASTICSEARCH_USERNAME = os.getenv('ELASTICSEARCH_USERNAME', 'admin')
ELASTICSEARCH_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD', "")
ENTITY_EMBEDDINGS_INDEX = os.getenv('ENTITY_EMBEDDINGS_INDEX', 'rag_entity_embeddings_index')
TEXT_ENTITY_EMBEDDINGS_INDEX = os.getenv('TEXT_ENTITY_EMBEDDINGS_INDEX', 'rag_text_entity_embeddings_index')

# Open AI
MAX_CHUNK_LENGTH = get_environment_variable('MAX_CHUNK_LENGTH', "10000")
GPT_CHAT_MODEL = get_environment_variable('GPT_CHAT_MODEL', 'gpt-3.5-turbo-1106')
RETRY_STOP = get_environment_variable('RETRY_STOP', '3')
EMBEDDING_MODEL = get_environment_variable('EMBEDDING_MODEL', 'text-embedding-ada-002')
