"""CLI script to launch the dashboard app."""

# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import from standard library
import argparse
import logging
from importlib.resources import files

# Import local functions
from .dashboard import build_app


# ==================================================================================================
# --- Launch app
# ==================================================================================================
def main(
    path_collider,
    port=8080,
    force_reload=False,
    ignore_footprint=False,
    simplify_tw=True,
    debug=False,
):

    # Log the initial configuration
    logging.info("Launching app with the following parameters:")
    logging.info(f"Collider path: {path_collider}")
    logging.info(f"Port: {port}")
    logging.info(f"Force reload: {force_reload}")
    logging.info(f"Ignore footprint: {ignore_footprint}")
    logging.info(f"Simplify Twiss tables: {simplify_tw}")
    logging.info(f"Debug: {debug}")

    # Build and run the app
    app, _ = build_app(
        path_collider,
        force_reload=force_reload,
        ignore_footprint=ignore_footprint,
        simplify_tw=simplify_tw,
    )  # server not needed for local deployment
    app.run_server(debug=debug, host="0.0.0.0", port=port)


if __name__ == "__main__":

    # Get the example collider path
    package_path = str(files("collider_dashboard"))
    example_collider_path = package_path + "/data/collider.json"

    # Get the collider path from the command line
    parser = argparse.ArgumentParser(
        prog="ColliderDashboard",
        description=(
            "A Dash application to visualize the observables and parameters of a collider built and"
            " configured with Xsuite."
        ),
        epilog="CC Colas Droin, 2023",
    )
    parser.add_argument(
        "-c",
        "--collider-path",
        type=str,
        help="Path of the collider to load.",
        required=False,
        default=example_collider_path,
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="Port to run the dashboard on.",
        required=False,
        default=8080,
    )
    parser.add_argument(
        "-f",
        "--force-reload",
        action="store_true",
        help="Force the update of the collider.",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--ignore-footprint",
        action="store_true",
        help="Ignore the footprint computation.",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--simplify",
        action="store_true",
        help="simplify the Twiss tables.",
        required=False,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Launch the app in debugging mode.",
        required=False,
    )

    args = parser.parse_args()
    collider_path = args.collider_path
    port = args.port
    force_reload = args.force_reload
    ignore_footprint = args.ignore_footprint
    simplify_tw = args.simplify
    debug = args.debug

    # Warn that the default collider is used if needed
    if collider_path == example_collider_path:
        logging.warning("No collider path was provided. Launching example collider.")

    # Launch the app
    main(
        collider_path,
        port=port,
        force_reload=force_reload,
        ignore_footprint=ignore_footprint,
        simplify_tw=simplify_tw,
        debug=debug,
    )
