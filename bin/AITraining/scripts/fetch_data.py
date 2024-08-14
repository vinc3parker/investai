import os
from flask import jsonify, send_file

def fetch(path):
    print(path)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        return jsonify({"error": "Model not found"}), 404