from transformers import pipeline
from logger import logger
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def rerank(results, query):
    try:
        logger.info("Re-ranking search results result : {results}".format(results=results))

        if not results:
            return []
        
        # Extract quote to be re-ranked
        texts = []
        for data in results:
            text = "Quote: " + data['quote'] + " Author: " + data['author'] + " Tags: " + str(data['tags'])
            texts.append(text)

        # Generate (query, document) pairs
        pairs = [[query, doc_text] for doc_text in texts]

        # Get scores for each (query, document) pair
        scores = cross_encoder.predict(pairs)

        # Attach the scores to the original results
        for i, score in enumerate(scores):
            results[i]['rerankScore'] = str(score)

        return results
    except Exception as e:
        logger.error(f"An error occurred while re-ranking results. {e}")
        raise Exception(e)
