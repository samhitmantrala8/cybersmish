from flask import Flask, request, jsonify
from transformers import pipeline

MODEL_ID = "samhitmantrala/smish_fin"
classifier = pipeline("text-classification", model=MODEL_ID)

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expect JSON: { "text": "..." }
    Returns JSON: { "label": "...", "adjusted_score": value, "raw": [...] }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "missing 'text' field"}), 400

    text = data["text"]
    try:
        result = classifier(text)
        if isinstance(result, list) and len(result) > 0:
            top = result[0]
            label = top.get("label")
            score = float(top.get("score"))

            if label.upper() == "NEGATIVE":
                adjusted_score = score
            else:  
                adjusted_score = 1 - score

            return jsonify({
                "label": label,
                "adjusted_score": adjusted_score,
                "raw": result
            }), 200
        else:
            return jsonify({"error": "unexpected model output", "raw": result}), 500
    except Exception as e:
        return jsonify({"error": "inference failure", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)