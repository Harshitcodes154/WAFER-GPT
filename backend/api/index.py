# api/index.py
import sys
from pathlib import Path

# Backend directory ko path mein add karein taaki llm.py aur model file directly load ho sakein
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.append(str(BACKEND_DIR))

from backend.main import app
