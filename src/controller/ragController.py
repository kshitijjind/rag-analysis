from flask import Blueprint, request
import uuid
from logger import logger
import time
import json
from src.service.searchingAndRetrieve import getRagAnalysisResponse

ragController = Blueprint('ragController', __name__)


@ragController.route('/rag/analysis', methods=['POST'])
def getRagAnalysis():
    try:
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        logger.info(f"Request to get RAG analysis request : {request} correlationId : {correlation_id}")

        response = getRagAnalysisResponse(request.json, correlation_id)
        logger.info(f"Request to get RAG analysis request : {time.time() - start_time} correlationId : {correlation_id}")

        # Return the response
        return response
    except Exception as e:
        logger.error(f"An error occurred while fetching RAG analysis. {e}")
        return json.dumps({"error": str(e)}), 500
