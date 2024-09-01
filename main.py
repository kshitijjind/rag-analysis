# Import module
from src.app import app
import os
from logger import logger

defaultPort = 5001

if __name__ == '__main__':
    logger.info("Starting the ragAnalysis server on port {}".format(defaultPort))
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', defaultPort)))