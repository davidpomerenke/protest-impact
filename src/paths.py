from pathlib import Path

import joblib

from src.types import NewsItem
from src.util.html import website_name

_root = Path(__file__).parent.parent.parent

external_data = _root / "data" / "external"
interim_data = _root / "data" / "interim"
processed_data = _root / "data" / "processed"
raw_data = _root / "data" / "raw"
figures = _root / "reports" / "figures"
tables = _root / "reports" / "tables"
models = _root / "models"
