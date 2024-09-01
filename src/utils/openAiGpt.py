from openai import OpenAI
from config import EMBEDDING_MODEL
from logger import logger
import config
from tenacity import (retry, stop_after_attempt, wait_random_exponential)

# Set up OpenAI API credentials
openai = OpenAI(api_key=config.OPENAI_API_KEY)


def create_embedding(data):
    try:
        embedding = openai.embeddings.create(
            input=data,
            model=EMBEDDING_MODEL,
        )

        return embedding.data[0].embedding
    except Exception as e:
        logger.error(f"An error occurred while creating embeddings. {e}")
        raise Exception(e)


def gpt_turbo_model(message, system_prompt, correlation_id, params=None):
    try:
        chunks = split_chunks(message, int(config.MAX_CHUNK_LENGTH))

        if len(chunks) > 1:
            logger.error(f"Input message was longer than max chunk size {chunks}, so picking the first chunk to "
                         f"response the query. correlationId: {correlation_id}")

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": chunks[0]}]

        logger.info(f"Query: {messages} correlationId: {correlation_id}")

        return completion_with_backoff(messages, correlation_id, config.GPT_CHAT_MODEL, params)

    except Exception as e:
        logger.error("An error occurred while getting the message reply from gptModel. ", e)
        raise Exception(e)


@retry(
    wait=wait_random_exponential(min=30, max=90),
    stop=stop_after_attempt(config.RETRY_STOP)
)
def completion_with_backoff(messages, correlationId, model, params=None):
    logger.info(f"Model selected: {model} params: {params} correlationId: {correlationId}")

    # Set default params
    if params is None: params = {}

    is_json = params.get('isJson', False)

    completion_params = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0.6,
    }

    if is_json:
        completion_params["response_format"] = {"type": "json_object"}

    completion = openai.chat.completions.create(**completion_params)

    return completion


def split_chunks(message, chunk_size):
    return [message[i:i + chunk_size] for i in range(0, len(message), chunk_size)]
