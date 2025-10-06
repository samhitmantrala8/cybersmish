from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

MODEL_ID = "samhitmantrala/smish_fin"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

# Apply dynamic quantization (reduces model size & RAM usage)
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Create the HuggingFace pipeline using the quantized model
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)  # device=-1 for CPU

# Flask app
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
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

            # Adjust score logic
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
