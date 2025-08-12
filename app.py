import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Rainfall Prediction App", page_icon="🌦️", layout="centered")

# -------- Utility: extract usable estimator from various saved formats --------
def extract_estimator(obj):
    # If obj itself is an estimator/pipeline
    if hasattr(obj, "predict"):
        return obj
    # If it's a dict, try common keys
    if isinstance(obj, dict):
        for key in ["model", "estimator", "pipeline", "clf", "classifier"]:
            if key in obj and hasattr(obj[key], "predict"):
                return obj[key]
    return None

@st.cache_resource
def load_model_from_file(path: str):
    try:
        loaded = joblib.load(path)
        est = extract_estimator(loaded)
        if est is None:
            st.error(
                "❌ Loaded file does not contain a valid estimator/pipeline. "
                "Expect a trained model or a dict with key 'model'."
            )
            return None
        return est
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"❌ Failed to load model from file: {e}")
        return None

def load_model_from_upload(uploaded_file):
    try:
        loaded = joblib.load(uploaded_file)
        est = extract_estimator(loaded)
        if est is None:
            st.error("❌ Uploaded file does not contain a valid estimator/pipeline.")
            return None
        return est
    except Exception as e:
        st.error(f"❌ Failed to load uploaded model: {e}")
        return None

# -------- UI --------
st.title("🌦️ Rainfall Prediction App (Classifier)")
st.write("Enter weather conditions to predict whether it will rain or not.")

# Try to load local model first
MODEL_PATH = "rainfall_prediction_model.pkl"
model = load_model_from_file(MODEL_PATH)

# If no local model, allow upload
if model is None:
    st.info(
        "No valid local model found. Upload a .pkl/.joblib containing a trained "
        "sklearn estimator/pipeline, or a dict with key 'model'."
    )
    uploaded = st.file_uploader("Upload model file", type=["pkl", "joblib"])
    if uploaded is not None:
        model = load_model_from_upload(uploaded)

# Input fields
col1, col2 = st.columns(2)
with col1:
    pressure = st.number_input(
        "Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1013.0, step=0.1
    )
    dewpoint = st.number_input(
        "Dew Point (°C)", min_value=-50.0, max_value=50.0, value=10.0, step=0.1
    )
    humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=60, step=1)
    cloud = st.slider("Cloud Cover (%)", min_value=0, max_value=100, value=50, step=1)
with col2:
    sunshine = st.number_input(
        "Sunshine Duration (hours)", min_value=0.0, max_value=24.0, value=5.0, step=0.1
    )
    winddirection = st.slider(
        "Wind Direction (°)", min_value=0, max_value=360, value=180, step=1
    )
    windspeed = st.number_input(
        "Wind Speed (km/h)", min_value=0.0, max_value=200.0, value=10.0, step=0.1
    )

# Predict Button
if st.button("Predict Rainfall"):
    if model is None:
        st.warning("⚠️ No model loaded. Please provide a valid model file.")
    else:
        # Build input DataFrame
        input_data = pd.DataFrame([{
            "pressure": pressure,
            "dewpoint": dewpoint,
            "humidity": humidity,
            "cloud": cloud,
            "sunshine": sunshine,
            "winddirection": winddirection,
            "windspeed": windspeed
        }])

        st.subheader("✅ Input Data")
        st.dataframe(input_data, use_container_width=True)

        try:
            # Ensure correct column order if your model expects it:
            # feature_order = ["pressure","dewpoint","humidity","cloud","sunshine","winddirection","windspeed"]
            # input_data = input_data[feature_order]

            pred = model.predict(input_data)
            y_hat = int(pred[0]) if hasattr(pred, "__len__") else int(pred)

            st.subheader("📈 Prediction")
            label = "🌧️ Rain" if y_hat == 1 else "☀️ No Rain"
            st.success(f"Predicted: {label}")

            # Probabilities (if available)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0]
                if hasattr(model, "classes_"):
                    classes = list(model.classes_)
                    prob_map = {str(c): float(p) for c, p in zip(classes, proba)}
                    if set(classes) == {0, 1}:
                        st.write("🔍 Prediction Probabilities:")
                        st.write({
                            "No Rain (0)": round(prob_map.get("0", 0.0), 3),
                            "Rain (1)": round(prob_map.get("1", 0.0), 3)
                        })
                    else:
                        st.write("🔍 Prediction Probabilities (by class):")
                        st.write({str(k): round(v, 3) for k, v in prob_map.items()})
                else:
                    st.write("🔍 Prediction Probabilities:")
                    st.write({f"class_{i}": round(float(p), 3) for i, p in enumerate(proba)})
            else:
                st.info("Model does not provide probabilities (predict_proba not available).")

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")

# Footer tip
st.caption(
    "Tip: Save your trained sklearn Pipeline to include preprocessing so the app can "
    "accept raw inputs directly."
)
