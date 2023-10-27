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
app = build_app()
server = app.server  # Define server for gunicorn
