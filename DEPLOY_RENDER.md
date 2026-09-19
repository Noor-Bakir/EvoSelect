# Render deployment

Use Python 3.11.9. The dependency versions are pinned to wheels available for
Python 3.11 to avoid building pandas/numpy from source.

Build:
pip install -r requirements.txt

Start:
gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT app:app

Health check:
`/api/health`
