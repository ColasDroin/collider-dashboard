from .dashboard import build_app

# ==================================================================================================
# --- Launch app
# Execute with gunicorn dashboard.__main__:server -b :8080
# Run silently with nohup gunicorn dashboard.__main__:server -b :8080 &
# Kill all gunicorn instances with pkill gunicorn
# ==================================================================================================
app = build_app()
server = app.server  # Define server for gunicorn
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)
