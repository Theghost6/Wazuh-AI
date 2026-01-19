from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("./xss_detector_model")
model = AutoModelForSequenceClassification.from_pretrained("./xss_detector_model").to(device)
model.eval()

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        payload = request.args.get('payload') or (request.get_json() or {}).get('payload')
        
        if not payload:
            return jsonify({"error": "missing payload"}), 400
        
        inputs = tokenizer(
            payload,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()
        
        # 0=BENIGN, 1=XSS
        pred_class = 'XSS' if pred == 1 else 'BENIGN'
        risk_level = 'HIGH' if (pred == 1 and conf > 0.9) else 'MEDIUM' if pred == 1 else 'NONE'
        
        display = payload[: 60] + "..." if len(payload) > 60 else payload
        icon = "❌" if pred == 1 else "✅"
        print(f"[{pred_class}] {icon} {display} ({conf*100:.1f}%)")
        
        return jsonify({
            "prediction": pred_class,
            "confidence": float(conf),
            "risk_level": risk_level,
            "is_xss": pred == 1
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model":  "Transformer-based XSS Detector"})

if __name__ == '__main__':
    print("\n🚀 XSS Detection API (Transformer)")
    app.run(host='0.0.0.0', port=5000, debug=False)
