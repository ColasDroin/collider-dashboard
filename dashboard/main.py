"""Module to build the dashboard app and link the corresponding server, such that it can be executed 
from e.g. gunicorn."""
# ==================================================================================================
# --- Imports
# ==================================================================================================
from .dashboard import build_app

# ==================================================================================================
# --- Launch app
# Execute with gunicorn dashboard.main:server -b :8080
# Run silently with nohup gunicorn dashboard.main:server -b :8080 &
# Kill all gunicorn instances with pkill gunicorn
# ==================================================================================================
app, server = build_app()  # server needed for gunicorn
