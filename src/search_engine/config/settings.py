"""
Configuración simple de rutas para el motor de búsqueda.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "data"
RAW_DOCUMENTS_DIR = DATA_DIR / "raw" / "documents"
PROCESSED_DIR = DATA_DIR / "processed"
STOP_WORDS_FILE = DATA_DIR / "stop_words.txt"
