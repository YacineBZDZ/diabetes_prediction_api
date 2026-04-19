import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # required — lets GitHub Pages call this from a different origin

# ── Load model once at startup ───────────────────────────────────────────────
data = np.load("params.npz")
L = sum(1 for k in data.files if k.startswith("W"))
parameters = {}
for l in range(1, L + 1):
    parameters[f"W{l}"] = data[f"W{l}"]
    parameters[f"b{l}"] = data[f"b{l}"]

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ── Inference helpers (copied from your notebook) ───────────────────────────
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def predict_proba(X, parameters):
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        Z = parameters[f"W{l}"] @ A + parameters[f"b{l}"]
        A = relu(Z)
    ZL = parameters[f"W{L}"] @ A + parameters[f"b{L}"]
    AL = sigmoid(ZL)
    return float(AL[0, 0])

# ── Feature order must exactly match training ────────────────────────────────
BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]
CONTINUOUS_FEATURES = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age"]
FEATURE_ORDER = BINARY_FEATURES + CONTINUOUS_FEATURES  # 17 features total

@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()

    # Build feature vector in the right order
    try:
        raw = np.array([[float(body[f]) for f in FEATURE_ORDER]])
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400

    # Apply the same preprocessing as training
    # 1. log1p on MentHlth and PhysHlth
    for col_name in ["MentHlth", "PhysHlth"]:
        idx = FEATURE_ORDER.index(col_name)
        raw[0, idx] = np.log1p(raw[0, idx])

    # 2. MinMax scale continuous features only
    cont_indices = [FEATURE_ORDER.index(f) for f in CONTINUOUS_FEATURES]
    raw[0, cont_indices] = scaler.transform(raw[:, cont_indices])[0]

    X = raw.T   # shape (17, 1)
    prob = predict_proba(X, parameters)
    label = int(prob >= 0.4)   # use your trained threshold

    return jsonify({
        "probability": round(prob, 4),
        "prediction": label,
        "label": "Diabetes" if label == 1 else "No diabetes"
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)