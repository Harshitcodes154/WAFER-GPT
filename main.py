import sys
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Add backend directory to system path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "backend"))

# Import your configured FastAPI app from backend/main.py
from backend.main import app

# Mount frontend static files (HTML, CSS, JS) at the root URL
FRONTEND_DIR = ROOT_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
