from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = DATA_RAW_DIR / "houses.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET_COLUMN = "Price (in rupees)"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_directories() -> None:
    """Create required project directories if they do not exist."""
    for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
