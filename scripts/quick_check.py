# scripts/quick_check.py
# This script is designed to run a quick pipeline check for my IoT anomaly detection project.
# It verifies whether the Python environment is ready, the dataset is available,
# the trained model and encoders can be loaded, and whether the stream simulator
# and incremental learning components work correctly. On a Raspberry Pi,
# it can also check GPIO libraries for the sensor.

import os, sys
from pathlib import Path

STATUS_OK = "✅"
STATUS_FAIL = "❌"
STATUS_WARN = "⚠️"

def say(ok, msg):
    # Helper function to print consistent pass/fail messages
    print(f"{STATUS_OK if ok else STATUS_FAIL} {msg}")

# Define paths to dataset and models
ROOT = Path(__file__).resolve().parents[1]
DATA_CSV   = ROOT / "data" / "Train_Test_Network_dataset" / "train_test_network.csv"
SAMPLE_CSV = ROOT / "sample.csv"

# In this project the models are saved under scripts/models
DEFAULT_MODELS_DIR = ROOT / "scripts" / "models"

# For running on my Windows laptop, I also define absolute paths
WIN_MODEL_PATH = Path(r"C:\Users\saite\Desktop\IoT_Anomaly_Detection_Project\scripts\models\kan_model.joblib")
WIN_ENCODERS_PATH = Path(r"C:\Users\saite\Desktop\IoT_Anomaly_Detection_Project\scripts\models\kan_encoders.joblib")

def resolve_model_paths():
    """
    This function decides which model/encoder paths to use.
    Priority: (1) environment variables, (2) Windows absolute paths, (3) repo-relative.
    """
    env_model = os.environ.get("KAN_MODEL_PATH")
    env_enc   = os.environ.get("KAN_ENCODERS_PATH")
    if env_model and env_enc and Path(env_model).exists() and Path(env_enc).exists():
        return Path(env_model), Path(env_enc)
    if WIN_MODEL_PATH.exists() and WIN_ENCODERS_PATH.exists():
        return WIN_MODEL_PATH, WIN_ENCODERS_PATH
    return DEFAULT_MODELS_DIR / "kan_model.joblib", DEFAULT_MODELS_DIR / "kan_encoders.joblib"

def check_env():
    """Check if Python and the main libraries (pandas, sklearn, river) are installed."""
    try:
        import platform, pandas as pd
        print(f"Python: {platform.python_version()}")
        print("pandas:", pd.__version__)
        try:
            import sklearn; print("scikit-learn:", sklearn.__version__)
        except Exception as e:
            print(STATUS_WARN, "scikit-learn not available:", e)
        try:
            import river; print("river:", river.__version__)
        except Exception as e:
            print(STATUS_WARN, "river not available:", e)
        say(True, "Environment imports successful")
        return True
    except Exception as e:
        say(False, f"Environment imports failed: {e}")
        return False

def check_data():
    """Check if the dataset is present and has the expected structure."""
    try:
        import pandas as pd
        if not DATA_CSV.exists():
            raise FileNotFoundError(f"Dataset missing at {DATA_CSV}")
        df = pd.read_csv(DATA_CSV)
        print(f"Data shape: {df.shape}")
        print("Columns:", list(df.columns)[:10], "...")
        if "label" not in df.columns:
            raise ValueError("Dataset does not have a 'label' column")
        say(True, "Dataset is readable and has 'label'")
        return True, df
    except Exception as e:
        say(False, f"Dataset check failed: {e}")
        return False, None

def check_models(df):
    """Load the trained model and encoders, then try a one-shot prediction."""
    try:
        import joblib, pandas as pd
        model_path, enc_path = resolve_model_paths()
        print("Model path:", model_path)
        print("Encoders path:", enc_path)
        if not model_path.exists() or not enc_path.exists():
            raise FileNotFoundError(f"Model or encoders not found at:\n  {model_path}\n  {enc_path}")
        model    = joblib.load(model_path)
        encoders = joblib.load(enc_path)

        # Sanity checks
        if not hasattr(model, "predict"):
            raise TypeError("Model object has no .predict() method")
        if not isinstance(encoders, dict):
            raise TypeError("Encoders are not in dictionary format")

        # Encode categorical features before prediction
        X = df.drop(columns=["label"]).copy()
        for c in X.select_dtypes(include="object").columns:
            if c in encoders:
                X[c] = encoders[c].transform(X[c].astype(str))

        # Choose either sample.csv or the first row of the dataset
        if SAMPLE_CSV.exists():
            s = pd.read_csv(SAMPLE_CSV)
            s = s[X.columns]
        else:
            s = X.iloc[[0]].copy()

        pred = model.predict(s)[0]
        print("One-shot prediction result:", pred)
        say(True, "Model and encoders loaded successfully")
        return True
    except Exception as e:
        say(False, f"Model check failed: {e}")
        return False

def check_stream_simulator():
    """Make sure the stream simulator script can be imported without errors."""
    try:
        import importlib.util
        path = ROOT / "scripts" / "stream_simulator.py"
        spec = importlib.util.spec_from_file_location("stream_simulator", path)
        mod  = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        say(True, "stream_simulator.py imported correctly")
        return True
    except Exception as e:
        say(False, f"stream_simulator.py import failed: {e}")
        return False

def check_incremental():
    """Run a small incremental learning test using river."""
    try:
        from river import tree, metrics, stream
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        df = pd.read_csv(DATA_CSV)
        for c in df.select_dtypes(include="object").columns:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        X, y = df.drop(columns=["label"]), df["label"]
        ds = stream.iter_pandas(X, y)
        model, acc = tree.HoeffdingTreeClassifier(), metrics.Accuracy()
        for i, (x, y_true) in enumerate(ds):
            y_pred = model.predict_one(x)
            model.learn_one(x, y_true)
            if y_pred is not None: acc.update(y_true, y_pred)
            if i >= 199: break
        print("Incremental accuracy on 200 samples:", f"{acc.get():.3f}")
        say(True, "Incremental learning test passed")
        return True
    except Exception as e:
        say(False, f"Incremental learning test failed: {e}")
        return False

def check_pi_gpio():
    """If running on Raspberry Pi, check if GPIO libraries are available."""
    if "--pi" not in sys.argv:
        print(STATUS_WARN, "Skipping Pi GPIO check (use --pi flag to enable)")
        return True
    try:
        import board, digitalio, adafruit_dht
        say(True, "Raspberry Pi GPIO libraries detected")
        return True
    except Exception as e:
        say(False, f"GPIO library check failed: {e}")
        return False

def main():
    print("🔎 QUICK PIPELINE CHECK\nProject root:", ROOT)
    ok_env = check_env()
    ok_data, df = check_data()
    ok_models = check_models(df) if ok_data else False
    ok_stream = check_stream_simulator()
    ok_inc = check_incremental()
    ok_pi = check_pi_gpio()
    all_ok = all([ok_env, ok_data, ok_models, ok_stream, ok_inc, ok_pi])
    print("\n====================")
    print("OVERALL:", "✅ PASS" if all_ok else "❌ ISSUES FOUND")
    print("====================")
    if not all_ok:
        print("If something failed, I will review the error output and fix it.")

if __name__ == "__main__":
    main()
