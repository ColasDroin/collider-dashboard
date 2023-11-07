"""a Plotly Dash app used to explore collider observables."""
# ==================================================================================================
# --- Imports
# ==================================================================================================
import logging

# ==================================================================================================
# --- Package version
# ==================================================================================================
__version__ = "0.1.1"

# ==================================================================================================
# --- Customize logging
# ==================================================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ==================================================================================================
# --- First log
# ==================================================================================================
logging.info("Starting imports")
