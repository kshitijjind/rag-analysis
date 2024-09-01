from elasticsearch import Elasticsearch
from config import ELASTICSEARCH_DOMAIN_URL, ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD


def connect_to_elasticsearch():
    # Connect to the local Elasticsearch instance
    return Elasticsearch(
        hosts=[{"host": ELASTICSEARCH_DOMAIN_URL, "port": 9200, "scheme": "http"}],
        http_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
    )
