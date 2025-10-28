import numpy as np
import pandas as pd
import joblib
import tritonclient.http as httpclient

# === Load preprocessor ===
preprocessor = joblib.load("models/preprocessor.pkl")

# === Example raw input ===
df = pd.DataFrame([
    {"BAIRRO": "ALDEOTA", "AREA_EDIFICADA": 120.0, "NUMERO_PAVIMENTOS": 15},
    {"BAIRRO": "MEIRELES", "AREA_EDIFICADA": 85.0, "NUMERO_PAVIMENTOS": 10}
])

# Transform data
X_trans = preprocessor.transform(df)
if hasattr(X_trans, "toarray"):
    X_trans = X_trans.toarray()
X_trans = X_trans.astype(np.float32)

# === Send to Triton ===
client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = [httpclient.InferInput("input", X_trans.shape, "FP32")]
inputs[0].set_data_from_numpy(X_trans)

outputs = [httpclient.InferRequestedOutput("variable")]

response = client.infer(model_name="decision_tree", inputs=inputs, outputs=outputs)
preds = response.as_numpy("variable")

print("Predictions:", preds.ravel())
