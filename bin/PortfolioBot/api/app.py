from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('./predict/<ticker>', methods=['GET'])
def predict(ticker):
    return None

if __name__ == '__main__':
    app.run(debug=True, port=5002)