python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e . pytest matplotlib
.\.venv\Scripts\python.exe -m pytest -q
