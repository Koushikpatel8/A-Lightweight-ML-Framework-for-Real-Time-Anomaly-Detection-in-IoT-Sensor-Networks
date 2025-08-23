# scripts/quick_check.py
import os, sys, time, json, warnings
from pathlib import Path

STATUS_OK = "✅"
STATUS_FAIL = "❌"
STATUS_WARN = "⚠️"

def say(ok, msg):
    print(f"{STATUS_OK if ok else STATUS_FAIL} {msg}")

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[1]

# Keep your original data paths (adjust if your dataset lives elsewhere)
DATA_CSV   = ROOT / "data" / "Train_Test_Network_dataset" / "train_test_network.csv"
SAMPLE_CSV = ROOT / "sample.csv"

# Default models dir in this repo actually lives under scripts/models
DEFAULT_MODELS_DIR = ROOT / "scripts" / "models"

# Preferred absolute Windows paths (used when they exist)
WIN_MODEL_PATH = Path(r"C:\Users\saite\Desktop\IoT_Anomaly_Detection_Project\scripts\models\kan_model.joblib")
WIN_ENCODERS_PATH = Path(r"C:\Users\saite\Desktop\IoT_Anomaly_Detection_Project\scripts\models\kan_encoders.joblib")

def resolve_model_paths():
    """
    Use the Windows absolute paths if they exist (when running on your laptop),
    otherwise fall back to repo-relative scripts/models.
    Also allow env overrides: KAN_MODEL_PATH, KAN_ENCODERS_PATH.
    """
    # Env override, if provided
    env_model = os.environ.get("KAN_MODEL_PATH")
    env_enc   = os.environ.get("KAN_ENCODERS_PATH")
    if env_model and env_enc:
        mp, ep = Path(env_model), Path(env_enc)
        if mp.exists() and ep.exists():
            return mp, ep

    # Preferred Windows absolute paths
    if WIN_MODEL_PATH.exists() and WIN_ENCODERS_PATH.exists():
        return WIN_MODEL_PATH, WIN_ENCODERS_PATH

    # Fallback to repo-relative scripts/models
    model_path = DEFAULT_MODELS_DIR / "kan_model.joblib"
    enc_path   = DEFAULT_MODELS_DIR / "kan_encoders.joblib"
    return model_path, enc_path

def check_env():
    ok = True
    try:
        import platform
        print(f"Python: {platform.python_version()}")
        import pandas as pd; print("pandas:", pd.__version__)
        try:
            import sklearn; print("scikit-learn:", sklearn.__version__)
        except Exception as e:
            print(STATUS_WARN, "scikit-learn not available:", e)
        try:
            import river; print("river:", river.__version__)
        except Exception as e:
            print(STATUS_WARN, "river not available:", e)
        say(True, "Environment imports")
    except Exception as e:
        ok = False
        say(False, f"Environment imports failed: {e}")
    return ok

def check_data():
    ok = True
    try:
        import pandas as pd
        if not DATA_CSV.exists():
            raise FileNotFoundError(f"Missing dataset at {DATA_CSV}")
        df = pd.read_csv(DATA_CSV)
        print(f"Data shape: {df.shape}")
        print("Columns:", list(df.columns)[:10], "...")
        if "label" not in df.columns:
            raise ValueError("No 'label' column in dataset")
        say(True, "Dataset readable & has 'label'")
        return ok, df
    except Exception as e:
        say(False, f"Dataset check failed: {e}")
        return False, None

def check_models(df):
    ok = True
    try:
        import joblib
        model_path, enc_path = resolve_model_paths()
        print("Model path:", model_path)
        print("Encoders path:", enc_path)
        if not model_path.exists() or not enc_path.exists():
            raise FileNotFoundError(f"Model or encoders not found:\n  {model_path}\n  {enc_path}")

        model    = joblib.load(model_path)
        encoders = joblib.load(enc_path)

        # quick sanity
        if not hasattr(model, "predict"):
            raise TypeError("Loaded model has no .predict()")
        if not isinstance(encoders, dict):
            raise TypeError("Encoders object is not a dict")

        # Build a single encoded sample aligned to training features
        import pandas as pd
        X = df.drop(columns=["label"]).copy()
        obj_cols = list(X.select_dtypes(include="object").columns)
        missing_enc = [c for c in obj_cols if c not in encoders]
        if missing_enc:
            print(STATUS_WARN, f"Encoders missing for: {missing_enc}")
        for c in obj_cols:
            if c in encoders:
                X[c] = encoders[c].transform(X[c].astype(str))

        # Use either sample.csv if present or first row of dataset
        if SAMPLE_CSV.exists():
            s = pd.read_csv(SAMPLE_CSV)
            s = s[X.columns]  # will raise if mismatch
        else:
            s = X.iloc[[0]].copy()

        # Predict
        pred = model.predict(s)[0]
        print("One-shot prediction:", pred)
        say(True, "Model + encoders load & predict")
        return ok
    except Exception as e:
        say(False, f"Model check failed: {e}")
        return False

def check_stream_simulator():
    # Import-only smoke check (don’t run the long stream).
    try:
        import importlib.util
        path = ROOT / "scripts" / "stream_simulator.py"
        spec = importlib.util.spec_from_file_location("stream_simulator", path)
        mod  = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        say(True, "stream_simulator.py imports")
        return True
    except Exception as e:
        say(False, f"stream_simulator.py import failed: {e}")
        return False

def check_incremental():
    try:
        from river import tree, metrics, stream
        import pandas as pd
        df = pd.read_csv(DATA_CSV)
        from sklearn.preprocessing import LabelEncoder
        for c in df.select_dtypes(include="object").columns:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        X = df.drop(columns=["label"])
        y = df["label"]
        ds = stream.iter_pandas(X, y)
        model = tree.HoeffdingTreeClassifier()
        acc = metrics.Accuracy()
        for i, (x, y_true) in enumerate(ds):
            y_pred = model.predict_one(x)
            model.learn_one(x, y_true)
            if y_pred is not None:
                acc.update(y_true, y_pred)
            if i >= 199:  # 200 samples smoke test
                break
        print("Incremental accuracy (200 samples, online):", f"{acc.get():.3f}")
        say(True, "Incremental learning (river) basic run")
        return True
    except Exception as e:
        say(False, f"Incremental learning check failed: {e}")
        return False

def check_pi_gpio():
    # Only runs on Pi when requested
    if "--pi" not in sys.argv:
        print(STATUS_WARN, "Skipping Pi GPIO check (run with --pi to enable)")
        return True
    try:
        import board, digitalio, adafruit_dht
        say(True, "Pi GPIO libs present (board/digitalio/adafruit_dht)")
        return True
    except Exception as e:
        say(False, f"Pi GPIO libs missing: {e}")
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
        print("If anything failed, copy/paste this output back to me and I’ll fix it.")

if __name__ == "__main__":
    main()
