import config
from src.common.constants import GENERATE_TEXT_PROMPT, SYSTEM_PROMPT
from src.database.elasticsearchConfig import connect_to_elasticsearch
from logger import logger
from src.dataset.loadDataset import read_pdf, create_chunks, loadDataset
from src.service.reRanking import rerank
from src.utils.openAiGpt import create_embedding, gpt_turbo_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

client = connect_to_elasticsearch()

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=10000)

dataset = loadDataset()


# Preprocess data
def combine_fields(item):
    return f"{item['text']}"


# Load dataset and TF-IDF vectorizer
combined_texts = [combine_fields(item) for item in dataset]
tfidf_matrix = vectorizer.fit_transform(combined_texts)

# Load dataset and BM25
tokenized_corpus = [word_tokenize(text.lower()) for text in combined_texts]
bm25 = BM25Okapi(tokenized_corpus)


def getRagAnalysisResponse(request, correlation_id):
    try:
        logger.info(f"Fetching RAG analysis for request : {request} correlationId : {correlation_id}")
        query = request.get("query")

        # Search for similar embeddings in Elasticsearch, then rerank and generate text using GPT from the top result
        embedding_search = search_es_embeddings(query)
        embedding_search_rank = rerank(embedding_search, query)
        embedding_search_output = generate_text_from_gpt(query, embedding_search_rank, correlation_id)
        embedding_precision, embedding_recall = calculate_precision_recall(embedding_search_rank, embedding_search)

        # Search for similar text in Elasticsearch using fuzzy matching, then rerank and generate text using GPT from
        # the top result
        fuzzy_search = search_es_fuzzy(query)
        fuzzy_search_rank = rerank(fuzzy_search, query)
        fuzzy_search_output = generate_text_from_gpt(query, fuzzy_search_rank, correlation_id)
        fuzzy_precision, fuzzy_recall = calculate_precision_recall(embedding_search_rank, embedding_search)

        # Search for similar text using TF-IDF, then rerank and generate text using GPT from the top result
        tfidf_search = tfidf_vectorizer(query)
        tfidf_search_rank = rerank(tfidf_search, query)
        tfidf_search_output = generate_text_from_gpt(query, tfidf_search_rank, correlation_id)
        tfidf_precision, tfidf_recall = calculate_precision_recall(embedding_search_rank, embedding_search)

        # Search for similar text using BM25, then rerank and generate text using GPT from the top result
        bm25_search = bm25_vectorizer(query)
        bm25_search_rank = rerank(bm25_search, query)
        bm25_search_output = generate_text_from_gpt(query, bm25_search_rank, correlation_id)
        bm25_precision, bm25_recall = calculate_precision_recall(embedding_search_rank, embedding_search)

        return {
            "embedding_search": {"results": embedding_search_rank, "output": embedding_search_output, "precision": embedding_precision, "recall": embedding_recall},
            "fuzzy_search": {"results": fuzzy_search_rank, "output": fuzzy_search_output, "precision": fuzzy_precision, "recall": fuzzy_recall},
            "tfidf_search": {"results": tfidf_search_rank, "output": tfidf_search_output, "precision": tfidf_precision, "recall": tfidf_recall},
            "bm25_search": {"results": bm25_search_rank, "output": bm25_search_output, "precision": bm25_precision, "recall": bm25_recall}
        }
    except Exception as e:
        logger.error(f"An error occurred while fetching RAG analysis. {e}")
        raise Exception(e)


def search_es_embeddings(search_request):
    """
        Searches for similar embeddings in Elasticsearch.

        Args:
            search_request : The request to search for in Elasticsearch.

        Returns:
            documents : A list of similar embeddings found in Elasticsearch.
    """
    try:
        embeddings = create_embedding(search_request)
        # Search for similar embeddings in Elasticsearch
        query = {
            "size": 5,
            "_source": ["text"],
            "knn": {
                "field": "vector",
                "query_vector": embeddings,
                "k": 5,
                "num_candidates": 5
            }
        }
        response = client.search(index=config.TEXT_ENTITY_EMBEDDINGS_INDEX, body=query)
        results = []
        for hit in response["hits"]["hits"]:
            results.append(hit["_source"])
        return results
    except Exception as e:
        logger.error(f"An error occurred while searching for similar embeddings in Elasticsearch. {e}")
        raise Exception(e)


def generate_text_from_gpt(query, description, correlation_id):
    try:

        # If no description found, return
        if not description or len(description) == 0:
            return

        # Extract description from the top result
        description = description[0]

        logger.info(
            f"Generating text from GPT for query : {query}, description : {description} correlationId : {correlation_id}")

        # Generate text using GPT from the query and description
        prompt = GENERATE_TEXT_PROMPT.format(**{"user_query": query, "description": description})

        # Get response from GPT
        ai_response = gpt_turbo_model(prompt, SYSTEM_PROMPT, correlation_id)

        # Extract the response from GPT
        ai_response = ai_response.choices[0].message.content

        logger.info(f"OpenAI response: {ai_response}, correlation_id: {correlation_id}")

        return ai_response
    except Exception as e:
        logger.error(f"An error occurred while generating text from GPT. {e}")
        raise Exception(e)


def search_es_fuzzy(search_request):
    """
        Searches for similar text in Elasticsearch using fuzzy matching.

        Args:
            search_request : The request to search for in Elasticsearch.

        Returns:
            documents : A list of similar text found in Elasticsearch.
    """
    try:
        # Search for similar text in Elasticsearch using fuzzy matching
        query = {
            "size": 5,
            "_source": ["text"],
            "query": {
                "multi_match": {
                    "fields": ["text"],
                    "query": search_request,
                    "fuzziness": "AUTO"
                }
            }
        }
        response = client.search(index=config.TEXT_ENTITY_EMBEDDINGS_INDEX, body=query)
        results = []
        for hit in response["hits"]["hits"]:
            results.append(hit["_source"])
        return results
    except Exception as e:
        logger.error(f"An error occurred while searching for similar text in Elasticsearch using fuzzy matching. {e}")
        raise Exception(e)


def tfidf_vectorizer(search_request, top_n=3):
    """
    Vectorizes the search request using TF-IDF and returns the top N results.
    :param search_request: dict with 'text'
    :param top_n: number of top results to return
    :return: list of dictionaries with the top N matching text and scores
    """
    try:
        logger.info(f"Vectorizing search request using TF-IDF : {search_request}")

        # Transform the search request using the fitted vectorizer
        search_vector = vectorizer.transform([search_request])

        # Compute cosine similarity between the search vector and the data matrix
        similarities = cosine_similarity(search_vector, tfidf_matrix).flatten()

        # Get the indices of the text sorted by similarity
        ranked_indices = similarities.argsort()[::-1]

        # Select the top N indices
        top_indices = ranked_indices[:top_n]

        # Return the top N matching text and their scores
        return [{
            'text': dataset[index]['text'],
            'score': similarities[index]
        } for index in top_indices]

    except Exception as e:
        logger.error(f"An error occurred while vectorizing search request using TF-IDF. {e}")
        raise Exception(e)


def bm25_vectorizer(search_request, top_n=3):
    """
    Vectorizes the search request using BM25 and returns the top N results.
    :param search_request: dict with 'text'
    :param top_n: number of top results to return
    :return: list of dictionaries with the top N matching text, authors, tags, and scores
    """
    try:
        logger.info(f"Vectorizing search request using BM25 : {search_request}")

        # Tokenize the search request
        tokenized_search = word_tokenize(search_request.lower())

        # Calculate BM25 scores for the search request
        scores = bm25.get_scores(tokenized_search)

        # Get the indices of the text sorted by score
        ranked_indices = scores.argsort()[::-1]

        # Select the top N indices
        top_indices = ranked_indices[:top_n]

        # Return the top N matching text, authors, tags, and their scores
        return [{
            'text': dataset[index]['text'],
            'score': scores[index]
        } for index in top_indices]

    except Exception as e:
        logger.error(f"An error occurred while vectorizing search request using BM25. {e}")
        raise Exception(e)


def calculate_precision_recall(retrieved_items, relevant_items):

    print(retrieved_items)
    print(relevant_items)
    # From dict retrieve list of text
    retrieved_items = [item['text'] for item in retrieved_items]
    relevant_items = [item['text'] for item in relevant_items]


    # Convert lists to sets for easier computation
    retrieved_set = set(retrieved_items)
    relevant_set = set(relevant_items)

    true_positives = len(retrieved_set.intersection(relevant_set))
    false_positives = len(retrieved_set - relevant_set)
    false_negatives = len(relevant_set - retrieved_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall
