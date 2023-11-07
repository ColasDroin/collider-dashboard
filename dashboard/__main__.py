"""CLI script to launch the dashboard app."""

# ==================================================================================================
# --- Imports
# ==================================================================================================
import argparse

from .dashboard import build_app


# ==================================================================================================
# --- Launch app
# ==================================================================================================
def main(path_collider, port):
    app, _ = build_app(path_collider)  # server not needed for local deployment
    app.run_server(debug=False, host="0.0.0.0", port=port)


if __name__ == "__main__":
    # Get the collider path from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", help="path to the collider")
    parser.add_argument("-port", help="port to run the dashboard on")
    args = parser.parse_args()
    path_collider = args.path
    port = args.port

    if path_collider is None:
        print("No collider path was provided. Launching example collider.")
        path_collider = "dashboard/data/collider.json"
    if port is None:
        print("No port was provided. Launching on port 8080.")
        port = 8080

    main(path_collider, port)
