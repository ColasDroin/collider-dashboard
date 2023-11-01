"""Module to build the dashboard app."""
# ==================================================================================================
# --- Imports
# ==================================================================================================
# Import from standard library
import logging

# Import third-party packages
from dash import Dash

# Import local functions
from .backend import init
from .callbacks import all_callbacks
from .layout import return_app_layout

# ==================================================================================================
# --- Load global variables and build app
# ==================================================================================================


def build_app(path_collider="/data/collider.json"):
    # Load dashboard variables
    dic_without_bb, dic_with_bb = init.init_from_collider(
        path_collider, load_global_variables_from_pickle=True
    )

    #################### App ####################
    logging.info("Defining app")
    app = Dash(
        __name__,
        title="Dashboard for current simulation",
        external_scripts=[
            "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
        ],
        suppress_callback_exceptions=False,
    )
    logging.info("Defining app layout")
    app.layout = return_app_layout()
    all_callbacks(app, dic_with_bb, dic_without_bb, path_collider)
    return app
