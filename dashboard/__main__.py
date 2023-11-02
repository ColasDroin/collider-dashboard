"""CLI script to launch the dashboard app."""

# ==================================================================================================
# --- Imports
# ==================================================================================================
import sys

from .dashboard import build_app


# ==================================================================================================
# --- Launch app
# ==================================================================================================
def main(path_collider):
    app, _ = build_app(path_collider)  # server not needed for local deployment
    app.run_server(debug=False, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    # Get the collider path from the command line
    if len(sys.argv) == 2:
        path_collider = sys.argv[1]
    else:
        print("No collider path was provided. Launching example collider.")
        path_collider = None

    main(path_collider)
