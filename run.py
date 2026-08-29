"""
1-Click Application Launcher
----------------------------
Orchestrates environment verification, directory initialization,
database seeding, ML model verification, and launches the FastAPI/Uvicorn server.
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

# Add project root to path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def check_dependencies():
    """Verifies that all critical dependencies are present."""
    required = ["fastapi", "uvicorn", "cv2", "numpy", "scipy", "sklearn", "sqlalchemy", "joblib"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n[ERROR] Missing required Python packages: {missing}")
        print(f"Please install dependencies by running:\n  pip install -r requirements.txt\n")
        sys.exit(1)


def bootstrap_environment():
    """Ensures database, models, and sample data are ready."""
    print("=" * 60)
    print("  AI-Powered Image Quality & Defect Detection System")
    print("=" * 60)

    # 1. Ensure runtime directories
    upload_dir = WORKSPACE_ROOT / "uploads"
    sample_dir = WORKSPACE_ROOT / "sample_images"
    ml_model_path = WORKSPACE_ROOT / "backend" / "app" / "ml" / "quality_model.pkl"

    upload_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 2. Verify ML Model exists; if not, train it automatically
    if not ml_model_path.exists():
        print("\n[ML] Trained model artifact not found. Initializing dataset and training...")
        from ml.generate_dataset import generate_dataset
        from ml.extract_features import extract_split_features
        from ml.train import train_model
        from ml.evaluate import evaluate_model

        base_ml = WORKSPACE_ROOT / "ml" / "dataset"
        generate_dataset(base_ml / "train", num_base_samples=50, start_seed=100)
        generate_dataset(base_ml / "test", num_base_samples=20, start_seed=500)
        extract_split_features(base_ml / "train", base_ml / "train_features.csv")
        extract_split_features(base_ml / "test", base_ml / "test_features.csv")
        train_model()
        evaluate_model()
    else:
        print("[ML] Model artifact verified: quality_model.pkl")

    # 3. Verify Database and Sample Data
    db_file = WORKSPACE_ROOT / "backend" / "inspection_system.db"
    if not db_file.exists():
        print("[DB] Initializing SQLite database and seeding demo records...")
        from scripts.init_db import init_and_seed_database
        init_and_seed_database()
    else:
        print("[DB] Database verified: inspection_system.db")


def main():
    check_dependencies()
    bootstrap_environment()

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    url = f"http://{host}:{port}"

    print(f"\n[SERVER] Launching FastAPI Web Application on {url}")
    print(f"[DOCS]   Interactive Swagger API Documentation: {url}/docs")
    print(f"[DASH]   Web Dashboard: {url}/\n")

    # Auto-open browser
    try:
        webbrowser.open(url)
    except Exception:
        pass

    import uvicorn
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
