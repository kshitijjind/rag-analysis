from flask import Flask, jsonify
from src.controller.ragController import ragController

app = Flask(__name__)


@app.route('/health')
def health_check():
    return jsonify({"message": "ok"})


app.register_blueprint(ragController)
