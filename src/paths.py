from pathlib import Path

_root = Path(__file__).parent.parent

external_data = _root / "data" / "external"
interim_data = _root / "data" / "interim"
processed_data = _root / "data" / "processed"
raw_data = _root / "data" / "raw"
figures = _root / "report" / "figures"
tables = _root / "report" / "tables"
models = _root / "models"
